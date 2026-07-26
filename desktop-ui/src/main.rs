#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod cluster_client;

use commands::{AppState, get_cluster, get_nodes, get_models, set_coordinator_url};

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    tauri::Builder::default()
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            get_cluster,
            get_nodes,
            get_models,
            set_coordinator_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
