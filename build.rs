use std::path::Path;
use std::process::Command;

fn main() {
    let shaders = vec![
        "shader.vert",
        "shader.frag",
        "cubemap.vert",
        "cubemap.frag",
    ];

    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets").join("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for shader in shaders {
        let input_path = src_dir.join(shader);
        let output_path = Path::new(&out_dir).join(format!("{}.spv", shader));

        let output = Command::new("glslangValidator")
            .arg("-V")
            .arg(input_path.to_str().unwrap())
            .arg("-o")
            .arg(output_path.to_str().unwrap())
            .output()
            .expect("Failed to execute glslangValidator");

        if !output.status.success() {
            panic!(
                "glslangValidator failed with error: {}\n{}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout),
            );
        }

        println!("cargo:rerun-if-changed={}", input_path.to_str().unwrap());
    }
}
