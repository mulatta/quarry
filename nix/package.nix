{
  perSystem =
    {
      pkgs,
      lib,
      ...
    }:
    let
      src = lib.cleanSourceWith {
        src = lib.cleanSource ../.;
        filter =
          path: type:
          let
            baseName = baseNameOf path;
          in
          type == "directory"
          || (lib.hasSuffix ".rs" path)
          || baseName == "Cargo.toml"
          || baseName == "Cargo.lock";
      };

      # Prebuilt ONNX Runtime v1.24.2 — matches ort-sys 2.0.0-rc.12.
      # Wrapped with autoPatchelfHook so .so files find libstdc++ via rpath.
      mkOnnxruntime =
        {
          suffix ? "",
          hash,
          extraBuildInputs ? [ ],
          ignoreMissing ? [ ],
        }:
        pkgs.stdenv.mkDerivation {
          pname = "onnxruntime-bin";
          version = "1.24.2";

          src = pkgs.fetchzip {
            url = "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64${suffix}-1.24.2.tgz";
            inherit hash;
          };

          nativeBuildInputs = [ pkgs.autoPatchelfHook ];
          buildInputs = [ pkgs.stdenv.cc.cc.lib ] ++ extraBuildInputs;

          autoPatchelfIgnoreMissingDeps = ignoreMissing;

          installPhase = ''
            mkdir -p $out/lib
            cp -a lib/*.so* $out/lib/
          '';
        };

      onnxruntimeCpu = mkOnnxruntime {
        hash = "sha256-0HdOtHBC4R21G/fJFuDzHrt9CrQkk4Q1jY5M0UAciWI=";
      };

      onnxruntimeGpu = mkOnnxruntime {
        suffix = "-gpu";
        hash = "sha256-nNLAXwsg0pWbXymLslPyMDyQkUuPKiygpbngGjWoB3M=";
        extraBuildInputs = with pkgs.cudaPackages; [
          libcublas
          libcurand
          libcufft
          cudnn
          cuda_cudart
        ];
        # TensorRT provider .so files need libnvinfer/libnvonnxparser,
        # but we only use the CUDA provider — skip patching those.
        ignoreMissing = [
          "libnvinfer.so.10"
          "libnvonnxparser.so.10"
        ];
      };

      mkQuarryEtl =
        {
          features ? null,
          ort,
          extraBuildInputs ? [ ],
        }:
        pkgs.rustPlatform.buildRustPackage {
          pname = "quarry-etl";
          version = "0.1.0";

          inherit src;

          cargoHash = "sha256-IsCmL2XCPzxMSfQxM9esH/eFe/nH9RAL8zB2bxRdJD0=";

          buildFeatures = features;

          dontUseCmakeConfigure = true;

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            autoPatchelfHook
          ];

          buildInputs = [
            pkgs.openssl
            pkgs.stdenv.cc.cc.lib
            ort
          ]
          ++ extraBuildInputs;

          ORT_LIB_LOCATION = "${ort}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
        };
    in
    {
      packages =
        let
          default = mkQuarryEtl { ort = onnxruntimeCpu; };
        in
        {
          inherit default;
          "quarry-etl" = default;
          "quarry-etl-cuda" = mkQuarryEtl {
            features = [ "cuda" ];
            ort = onnxruntimeGpu;
          };
        };
    };
}
