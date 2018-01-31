# Nix development environment
#
# source <bbp-nixpkgs>/sourcethis.sh
#
# nix-build
# nix-shell
#
with import <nixpkgs> {};
{
  projectionizer = projectionizer.overrideDerivation (oldAtr: rec {
    version = "DEV_ENV";
    src = ./.;
  });
}
