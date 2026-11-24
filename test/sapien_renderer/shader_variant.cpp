#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <svulkan2/shader/glsl_compiler.h>

namespace fs = std::filesystem;
using svulkan2::GLSLCompiler;

namespace {

/** A compute shader whose stored value comes from an include and a preprocessor variant. */
class ShaderVariantTest : public ::testing::Test {
protected:
  void SetUp() override {
    GLSLCompiler::InitializeProcess();
    mDir = fs::temp_directory_path() /
           ("sapien_shader_variant_" + std::to_string(::testing::UnitTest::GetInstance()
                                                           ->current_test_info()
                                                           ->line()));
    fs::create_directories(mDir);
    write("value.glsl", "const float BASE = 1.0;\n");
    write("main.comp", "#version 450\n"
                       "#include \"value.glsl\"\n"
                       "layout(set = 0, binding = 0, r32f) uniform writeonly image2D outValue;\n"
                       "#ifndef SCALE\n"
                       "#define SCALE 1.0\n"
                       "#endif\n"
                       "void main() {\n"
                       "#ifdef FLIP\n"
                       "  imageStore(outValue, ivec2(0), vec4(-BASE * SCALE));\n"
                       "#else\n"
                       "  imageStore(outValue, ivec2(0), vec4(BASE * SCALE));\n"
                       "#endif\n"
                       "}\n");
  }

  void TearDown() override {
    fs::remove_all(mDir);
    GLSLCompiler::FinalizeProcess();
  }

  void write(std::string const &name, std::string const &text) {
    std::ofstream(mDir / name) << text;
  }

  std::vector<uint32_t> const &compile(svulkan2::ShaderDefines const &defines = {}) {
    return GLSLCompiler::compileGlslFileCached(vk::ShaderStageFlagBits::eCompute,
                                               mDir / "main.comp", defines);
  }

  fs::path mDir;
};

TEST_F(ShaderVariantTest, DefinesSelectDifferentCode) {
  auto plain = compile();
  auto flipped = compile({{"FLIP", ""}});
  auto scaled = compile({{"SCALE", "2.0"}});
  EXPECT_NE(plain, flipped);
  EXPECT_NE(plain, scaled);
  EXPECT_NE(flipped, scaled);
  // a define equal to the shader's own default changes nothing in the generated code
  EXPECT_EQ(plain, compile({{"SCALE", "1.0"}}));
}

TEST_F(ShaderVariantTest, RepeatedRequestsShareOneCacheEntry) {
  auto const &first = compile({{"FLIP", ""}});
  auto const &second = compile({{"FLIP", ""}});
  EXPECT_EQ(&first, &second);
}

TEST_F(ShaderVariantTest, EditedIncludeCompilesAgain) {
  auto before = compile();
  write("value.glsl", "const float BASE = 3.0;\n");
  EXPECT_NE(before, compile());
}

TEST_F(ShaderVariantTest, CompileErrorsNameTheSourceLine) {
  write("main.comp", "#version 450\nvoid main() {\n  undefined_symbol;\n}\n");
  try {
    compile({{"A", "1"}, {"B", "2"}});
    FAIL() << "expected a compile error";
  } catch (std::runtime_error const &error) {
    // the two inserted define lines must not shift the reported line
    EXPECT_NE(std::string(error.what()).find("main.comp:3"), std::string::npos) << error.what();
  }
}

TEST_F(ShaderVariantTest, DefinesNeedAVersionLine) {
  write("main.comp", "void main() {}\n");
  EXPECT_THROW(compile({{"A", ""}}), std::runtime_error);
}

} // namespace
