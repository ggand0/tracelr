use eframe::egui;

use crate::build_info::BuildInfo;
use crate::theme::UiTheme;

/// Show the about modal. Dismiss via Escape or click-outside.
pub fn show_about_modal(ctx: &egui::Context, show: &mut bool, theme: &UiTheme) {
    if !*show {
        return;
    }

    // Semi-transparent backdrop
    let screen = ctx.screen_rect();
    egui::Area::new(egui::Id::new("about_backdrop"))
        .fixed_pos(screen.min)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let response = ui.allocate_response(screen.size(), egui::Sense::click());
            ui.painter().rect_filled(screen, 0.0, theme.backdrop);
            if response.clicked() {
                *show = false;
            }
        });

    // Modal content (Tooltip order so it renders above the Foreground backdrop)
    egui::Area::new(egui::Id::new("about_modal"))
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .order(egui::Order::Tooltip)
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme.card_bg)
                .stroke(egui::Stroke::new(1.0, theme.card_stroke))
                .corner_radius(8.0)
                .inner_margin(20.0)
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new("tracelr")
                                .size(25.0)
                                .strong(),
                        );
                        ui.add_space(15.0);

                        ui.label(
                            egui::RichText::new(format!(
                                "Version {}",
                                BuildInfo::display_version()
                            ))
                            .size(15.0),
                        );

                        ui.label(
                            egui::RichText::new(format!(
                                "Commit: {}",
                                BuildInfo::git_hash_short()
                            ))
                            .size(12.0)
                            .color(theme.muted),
                        );

                        ui.label(
                            egui::RichText::new(format!(
                                "Platform: {}",
                                BuildInfo::target_platform()
                            ))
                            .size(12.0)
                            .color(theme.muted),
                        );

                        ui.add_space(8.0);

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Author: ").size(15.0));
                            ui.label(
                                egui::RichText::new("Gota Gando")
                                    .size(15.0)
                                    .color(theme.accent),
                            );
                        });

                        ui.add_space(4.0);

                        ui.label(egui::RichText::new("Learn more at:").size(15.0));
                        let link_text = "https://github.com/ggand0/tracelr";
                        if ui
                            .add(
                                egui::Label::new(
                                    egui::RichText::new(link_text)
                                        .size(16.0)
                                        .color(theme.accent),
                                )
                                .sense(egui::Sense::click()),
                            )
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .clicked()
                        {
                            let _ = webbrowser::open(link_text);
                        }
                    });
                });
        });

    // Escape to close
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        *show = false;
    }
}
