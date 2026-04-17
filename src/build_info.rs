pub struct BuildInfo;

#[allow(dead_code)]
impl BuildInfo {
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    pub fn build_timestamp() -> &'static str {
        env!("BUILD_TIMESTAMP")
    }

    pub fn git_hash_short() -> &'static str {
        env!("GIT_HASH_SHORT")
    }

    pub fn target_platform() -> &'static str {
        env!("TARGET_PLATFORM")
    }

    pub fn build_profile() -> &'static str {
        env!("BUILD_PROFILE")
    }

    pub fn display_version() -> String {
        format!("{} ({})", Self::version(), Self::build_timestamp())
    }
}
