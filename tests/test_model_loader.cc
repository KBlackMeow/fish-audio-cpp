#include "model/loader.h"
#include <gtest/gtest.h>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <vector>

namespace fish {
namespace {

// Helper: write a minimal .bin file and return its path.
// The file is created in the current directory and cleaned up in TearDown.
class BinFile {
public:
    struct Spec {
        std::string name;
        uint32_t dtype_val;
        uint32_t ndim;
        int64_t shape[8];
        uint64_t data_size;
    };

    BinFile(const std::string& filename) : path_(filename) {}

    // Build a valid .bin file from specs and return true on success.
    bool write(const std::vector<Spec>& specs) {
        std::ofstream out(path_, std::ios::binary);
        if (!out) return false;

        uint32_t num_tensors = static_cast<uint32_t>(specs.size());

        // Calculate data section offset (aligned to 256)
        uint64_t header_size = 4 + 4 + 4;  // magic + version + num_tensors
        header_size += num_tensors * sizeof(TensorHeader);
        uint64_t data_offset = ((header_size + 255) / 256) * 256;

        // Write file header
        uint32_t magic = MAGIC;
        uint32_t version = FORMAT_VERSION;
        out.write(reinterpret_cast<const char*>(&magic), 4);
        out.write(reinterpret_cast<const char*>(&version), 4);
        out.write(reinterpret_cast<const char*>(&num_tensors), 4);

        // Write tensor headers and track data position
        uint64_t current_data_pos = data_offset;
        for (const auto& spec : specs) {
            TensorHeader hdr{};
            std::strncpy(hdr.name, spec.name.c_str(), sizeof(hdr.name) - 1);
            hdr.dtype_val = spec.dtype_val;
            hdr.ndim = spec.ndim;
            std::memcpy(hdr.shape, spec.shape, sizeof(hdr.shape));
            hdr.data_offset = current_data_pos;
            hdr.data_size = spec.data_size;

            out.write(reinterpret_cast<const char*>(&hdr), sizeof(TensorHeader));
            current_data_pos += spec.data_size;
        }

        // Pad to data_offset
        uint64_t written_so_far = 4 + 4 + 4 + num_tensors * sizeof(TensorHeader);
        for (uint64_t i = written_so_far; i < data_offset; i++) {
            out.put('\0');
        }

        // Write dummy data for each tensor
        std::vector<char> dummy(4096, 'X');
        for (const auto& spec : specs) {
            uint64_t remaining = spec.data_size;
            while (remaining > 0) {
                uint64_t chunk = std::min<uint64_t>(remaining, dummy.size());
                out.write(dummy.data(), chunk);
                remaining -= chunk;
            }
        }

        return out.good();
    }

    const std::string& path() const { return path_; }

    ~BinFile() {
        std::remove(path_.c_str());
    }

private:
    std::string path_;
};

TEST(ModelLoaderTest, LoadValidBinFile) {
    BinFile bf("/tmp/test_model_loader_valid.bin");

    BinFile::Spec spec_a;
    spec_a.name = "weight_a";
    spec_a.dtype_val = static_cast<uint32_t>(DType::FP32);
    spec_a.ndim = 2;
    spec_a.shape[0] = 4;
    spec_a.shape[1] = 8;
    spec_a.data_size = 4 * 8 * 4;  // FP32

    BinFile::Spec spec_b;
    spec_b.name = "bias_b";
    spec_b.dtype_val = static_cast<uint32_t>(DType::FP16);
    spec_b.ndim = 1;
    spec_b.shape[0] = 16;
    spec_b.data_size = 16 * 2;  // FP16

    ASSERT_TRUE(bf.write({spec_a, spec_b}));

    ModelLoader loader;
    ASSERT_TRUE(loader.load(bf.path()));
    EXPECT_EQ(loader.num_tensors(), 2);

    // Check has()
    EXPECT_TRUE(loader.has("weight_a"));
    EXPECT_TRUE(loader.has("bias_b"));
    EXPECT_FALSE(loader.has("nonexistent"));

    // Check get() for weight_a
    TensorView view_a = loader.get("weight_a");
    EXPECT_EQ(view_a.dtype, DType::FP32);
    ASSERT_EQ(view_a.shape.size(), 2);
    EXPECT_EQ(view_a.shape[0], 4);
    EXPECT_EQ(view_a.shape[1], 8);
    EXPECT_EQ(view_a.numel(), 32);
    EXPECT_EQ(view_a.nbytes(), 128);
    EXPECT_NE(view_a.data, nullptr);

    // Check get() for bias_b
    TensorView view_b = loader.get("bias_b");
    EXPECT_EQ(view_b.dtype, DType::FP16);
    ASSERT_EQ(view_b.shape.size(), 1);
    EXPECT_EQ(view_b.shape[0], 16);
    EXPECT_EQ(view_b.numel(), 16);
    EXPECT_EQ(view_b.nbytes(), 32);
    EXPECT_NE(view_b.data, nullptr);
}

TEST(ModelLoaderTest, RejectsInvalidMagic) {
    // Write a file with wrong magic
    std::ofstream out("/tmp/test_model_loader_bad_magic.bin", std::ios::binary);
    uint32_t bad_magic = 0xDEADBEEF;
    out.write(reinterpret_cast<const char*>(&bad_magic), 4);
    out.close();

    ModelLoader loader;
    EXPECT_FALSE(loader.load("/tmp/test_model_loader_bad_magic.bin"));
    EXPECT_EQ(loader.num_tensors(), 0);

    std::remove("/tmp/test_model_loader_bad_magic.bin");
}

TEST(ModelLoaderTest, RejectsNonExistentFile) {
    ModelLoader loader;
    EXPECT_FALSE(loader.load("/tmp/does_not_exist_xyz.bin"));
    EXPECT_EQ(loader.num_tensors(), 0);
}

TEST(ModelLoaderTest, MoveSemantics) {
    BinFile bf("/tmp/test_model_loader_move.bin");

    BinFile::Spec spec;
    spec.name = "tensor_x";
    spec.dtype_val = static_cast<uint32_t>(DType::BF16);
    spec.ndim = 3;
    spec.shape[0] = 2;
    spec.shape[1] = 3;
    spec.shape[2] = 4;
    spec.data_size = 2 * 3 * 4 * 2;  // BF16 = 2 bytes

    ASSERT_TRUE(bf.write({spec}));

    ModelLoader loader1;
    ASSERT_TRUE(loader1.load(bf.path()));
    EXPECT_TRUE(loader1.has("tensor_x"));

    // Move construct
    ModelLoader loader2(std::move(loader1));
    EXPECT_TRUE(loader2.has("tensor_x"));
    EXPECT_EQ(loader2.num_tensors(), 1);
    // loader1 should be empty now
    EXPECT_EQ(loader1.num_tensors(), 0);

    // Move assign
    ModelLoader loader3;
    loader3 = std::move(loader2);
    EXPECT_TRUE(loader3.has("tensor_x"));
    EXPECT_EQ(loader3.num_tensors(), 1);
    EXPECT_EQ(loader2.num_tensors(), 0);
}

TEST(ModelLoaderTest, UnloadClearsState) {
    BinFile bf("/tmp/test_model_loader_unload.bin");

    BinFile::Spec spec;
    spec.name = "tensor_y";
    spec.dtype_val = static_cast<uint32_t>(DType::FP32);
    spec.ndim = 1;
    spec.shape[0] = 10;
    spec.data_size = 40;

    ASSERT_TRUE(bf.write({spec}));

    ModelLoader loader;
    ASSERT_TRUE(loader.load(bf.path()));
    EXPECT_EQ(loader.num_tensors(), 1);

    loader.unload();
    EXPECT_EQ(loader.num_tensors(), 0);
    EXPECT_FALSE(loader.has("tensor_y"));
    EXPECT_THROW(loader.get("tensor_y"), std::runtime_error);
}

TEST(ModelLoaderTest, DataPointersAreValid) {
    BinFile bf("/tmp/test_model_loader_data.bin");

    BinFile::Spec spec_a;
    spec_a.name = "data_tensor";
    spec_a.dtype_val = static_cast<uint32_t>(DType::FP32);
    spec_a.ndim = 2;
    spec_a.shape[0] = 2;
    spec_a.shape[1] = 2;
    spec_a.data_size = 16;  // 4 floats

    ASSERT_TRUE(bf.write({spec_a}));

    ModelLoader loader;
    ASSERT_TRUE(loader.load(bf.path()));

    TensorView view = loader.get("data_tensor");
    ASSERT_NE(view.data, nullptr);

    // Data pointers from the same loader should point within mapped region
    auto* chars = static_cast<char*>(view.data);
    // Should be readable (no segfault)
    volatile char c = chars[0];
    (void)c;
}

}  // namespace
}  // namespace fish
