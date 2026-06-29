#include "model/loader.h"
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <cstring>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace fish {

ModelLoader::~ModelLoader() {
    unload();
}

ModelLoader::ModelLoader(ModelLoader&& other) noexcept
    : fd_(other.fd_)
    , mapped_data_(other.mapped_data_)
    , mapped_size_(other.mapped_size_)
    , headers_(std::move(other.headers_))
    , data_section_offset_(other.data_section_offset_)
{
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.mapped_size_ = 0;
}

ModelLoader& ModelLoader::operator=(ModelLoader&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) unload();
        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        mapped_size_ = other.mapped_size_;
        headers_ = std::move(other.headers_);
        data_section_offset_ = other.data_section_offset_;
        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.mapped_size_ = 0;
    }
    return *this;
}

bool ModelLoader::load(const std::string& bin_path) {
    if (fd_ >= 0) unload();

#ifdef _WIN32
    HANDLE hFile = CreateFileA(bin_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        spdlog::error("Failed to open model file: {}", bin_path);
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        spdlog::error("Failed to get file size: {}", bin_path);
        CloseHandle(hFile);
        return false;
    }
    mapped_size_ = static_cast<size_t>(fileSize.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        spdlog::error("Failed to create file mapping: {}", bin_path);
        CloseHandle(hFile);
        return false;
    }

    mapped_data_ = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);   // mapping handle no longer needed after MapViewOfFile
    CloseHandle(hFile);

    if (!mapped_data_) {
        spdlog::error("Failed to map view of file: {}", bin_path);
        return false;
    }
    fd_ = 1;  // mark as loaded (Windows: no fd, use sentinel)
#else
    fd_ = open(bin_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        spdlog::error("Failed to open model file: {}", bin_path);
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        spdlog::error("Failed to stat model file");
        close(fd_);
        fd_ = -1;
        return false;
    }
    mapped_size_ = st.st_size;

    mapped_data_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        spdlog::error("Failed to mmap model file");
        close(fd_);
        fd_ = -1;
        return false;
    }
#endif

    // Parse header
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_);

    uint32_t magic;
    std::memcpy(&magic, ptr, 4);
    ptr += 4;
    if (magic != MAGIC) {
        spdlog::error("Invalid magic: 0x{:08X}, expected 0x{:08X}", magic, MAGIC);
        unload();
        return false;
    }

    uint32_t version;
    std::memcpy(&version, ptr, 4);
    ptr += 4;
    if (version != FORMAT_VERSION) {
        spdlog::error("Unsupported version: {}", version);
        unload();
        return false;
    }

    uint32_t num_tensors;
    std::memcpy(&num_tensors, ptr, 4);
    ptr += 4;

    spdlog::info("Loading {} tensors from {}", num_tensors, bin_path);

    for (uint32_t i = 0; i < num_tensors; i++) {
        TensorHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(TensorHeader));
        ptr += sizeof(TensorHeader);

        std::string name(hdr.name);
        headers_[name] = hdr;

        spdlog::debug("  [{}] {} shape[{}] dtype={} size={}",
                      i, name, hdr.ndim, hdr.dtype_val, hdr.data_size);
    }

    data_section_offset_ = ptr - static_cast<const uint8_t*>(mapped_data_);
    spdlog::info("Loaded {} tensors, data section at offset {}",
                 num_tensors, data_section_offset_);
    return true;
}

TensorView ModelLoader::get(const std::string& name) const {
    auto it = headers_.find(name);
    if (it == headers_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const TensorHeader& hdr = it->second;
    TensorView view;
    view.data = static_cast<char*>(mapped_data_) + hdr.data_offset;
    view.dtype = static_cast<DType>(hdr.dtype_val);
    view.shape.assign(hdr.shape, hdr.shape + hdr.ndim);
    return view;
}

bool ModelLoader::has(const std::string& name) const {
    return headers_.count(name) > 0;
}

bool ModelLoader::pin_memory() {
    if (!mapped_data_) {
        spdlog::error("pin_memory: no mapped data");
        return false;
    }
    cudaError_t err = cudaHostRegister(mapped_data_, mapped_size_,
                                       cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        spdlog::error("pin_memory failed: {} ({} MB)",
                      cudaGetErrorString(err), mapped_size_ / (1024*1024));
        return false;
    }
    spdlog::info("Pinned {} MB for GPU access", mapped_size_ / (1024*1024));
    return true;
}

void ModelLoader::unload() {
    if (mapped_data_) {
        cudaHostUnregister(mapped_data_);
#ifdef _WIN32
        UnmapViewOfFile(mapped_data_);
#else
        munmap(mapped_data_, mapped_size_);
#endif
        mapped_data_ = nullptr;
    }
    fd_ = -1;
    headers_.clear();
    mapped_size_ = 0;
}

}  // namespace fish
