{ inputs, ... }:
{
  perSystem =
    {
      pkgs,
      lib,
      ...
    }:
    let
      craneLib = inputs.crane.mkLib pkgs;

      src = lib.fileset.toSource {
        root = ../.;
        fileset = lib.fileset.unions [
          ../Cargo.toml
          ../Cargo.lock
          (craneLib.fileset.commonCargoSources ../core)
          (craneLib.fileset.commonCargoSources ../cli)
        ];
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

      # Two-phase crane build: deps are cached separately from source.
      # Each variant (CPU/CUDA) gets its own cargoArtifacts because
      # ORT_LIB_LOCATION and features differ → different derivation inputs.
      mkQuarryEtl =
        {
          features ? [ ],
          ort,
          extraBuildInputs ? [ ],
        }:
        let
          featureArgs = lib.optionalString (
            features != [ ]
          ) " --features ${lib.concatStringsSep "," features}";

          commonArgs = {
            inherit src;
            pname = "quarry-etl";
            version = "0.1.0";
            strictDeps = true;
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

            cargoExtraArgs = "-p quarry-etl" + featureArgs;
            ORT_LIB_LOCATION = "${ort}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
          };

          # Phase 1: compile dependencies only (cached until Cargo.lock changes)
          cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        in
        # Phase 2: compile project source (reuses cached deps)
        craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            doCheck = false;
          }
        );
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
