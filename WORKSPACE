workspace(name = "opencv_test")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

new_local_repository(
    name = "opencv",
    build_file = "@//third_party:opencv.BUILD",
    path = "/usr/include",
)

http_archive(
    name ="onnxruntime",
    build_file = "//third_party:onnxruntime.BUILD",
    sha256 = "70c769771ad4b6d63b87ca1f62d3f11e998ea0b9d738d6bbdd6a5e6d8c1deb31",
    strip_prefix = "onnxruntime-linux-x64-1.16.2",
    urls = [
        "https://github.com/microsoft/onnxruntime/releases/download/v1.16.2/onnxruntime-linux-x64-1.16.2.tgz"
    ]
)

git_repository(
    name = "yaml-cpp",  # 仓库在 Bazel 中的名称
    remote = "https://github.com/jbeder/yaml-cpp.git",  # Git URL
    tag = "yaml-cpp-0.7.0",
)

git_repository(
    name = "zlib",  # 仓库在 Bazel 中的名称
    remote = "https://github.com/madler/zlib.git",  # Git URL
    commit = "5a82f71ed1dfc0bec044d9702463dbdf84ea3b71"
)

new_local_repository(
    name = "polyclipping",
    build_file = "@//third_party:polyclipping.BUILD",
    path = "/usr/include",
)

http_archive(
    name = "hedron_compile_commands",

    # Replace the commit hash (0e990032f3c5a866e72615cf67e5ce22186dcb97) in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/0e990032f3c5a866e72615cf67e5ce22186dcb97.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-0e990032f3c5a866e72615cf67e5ce22186dcb97",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
hedron_compile_commands_setup_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive_transitive()
