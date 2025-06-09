use std::path::PathBuf;
use std::process::Command;

fn main() {
    let shaders = [
        ("shader.glsl.vert", "shader.spirv.vert"),
        ("shader.glsl.frag", "shader.spirv.frag"),
    ];

    for (input, output) in shaders {
        let shaders_dir = PathBuf::from("./shaders/");
        let input_path = shaders_dir.join(input);
        let output_path = shaders_dir.join(output);

        let status = Command::new("glslc")
            .arg(input_path)
            .arg("-o")
            .arg(output_path)
            .status()
            .unwrap();

        if !status.success() {
            panic!("Failed to compile shader: {input}");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    for (input, output) in shaders {
        println!("cargo:rerun-if-changed=shaders/{input}");
        println!("cargo:rerun-if-changed=shaders/{output}");
    }
}
