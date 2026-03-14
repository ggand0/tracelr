use eframe::egui;

/// Centralized color theme for all custom UI elements.
#[allow(dead_code)] // fields used in settings/about modals (Phase 4)
pub struct UiTheme {
    pub accent: egui::Color32,
    pub backdrop: egui::Color32,
    pub card_bg: egui::Color32,
    pub card_stroke: egui::Color32,
    pub section_bg: egui::Color32,
    pub heading: egui::Color32,
    pub muted: egui::Color32,
    pub toggle_off: egui::Color32,
    pub toggle_knob: egui::Color32,
    pub menu_hover: egui::Color32,
}

impl UiTheme {
    /// Teal dark theme matching the iced ViewSkater version.
    pub fn teal_dark() -> Self {
        Self {
            accent: egui::Color32::from_rgb(26, 189, 208),
            backdrop: egui::Color32::from_black_alpha(140),
            card_bg: egui::Color32::from_gray(40),
            card_stroke: egui::Color32::from_gray(80),
            section_bg: egui::Color32::from_gray(30),
            heading: egui::Color32::from_gray(180),
            muted: egui::Color32::from_gray(140),
            toggle_off: egui::Color32::from_gray(50),
            toggle_knob: egui::Color32::from_gray(240),
            menu_hover: egui::Color32::from_gray(60),
        }
    }

    /// Apply the theme to egui's built-in widget visuals.
    pub fn apply_to_visuals(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();

        style.visuals.selection.bg_fill = self.accent;
        style.visuals.hyperlink_color = self.accent;
        style.visuals.widgets.active.bg_fill = self.accent;

        style.visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_gray(210);
        style.visuals.widgets.inactive.fg_stroke.color = egui::Color32::from_gray(220);
        style.visuals.widgets.hovered.fg_stroke.color = egui::Color32::from_gray(255);
        style.visuals.widgets.active.fg_stroke.color = egui::Color32::from_gray(255);
        style.visuals.widgets.open.fg_stroke.color = egui::Color32::from_gray(255);

        style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
        style.visuals.popup_shadow = egui::Shadow::NONE;
        style.visuals.window_shadow = egui::Shadow::NONE;

        style.visuals.widgets.hovered.weak_bg_fill = self.menu_hover;
        style.visuals.widgets.open.weak_bg_fill = self.menu_hover;

        ctx.set_style(style);
    }
}
