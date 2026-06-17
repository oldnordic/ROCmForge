use super::cli::Args;

fn should_handle_server_cli(args: &Args) -> bool {
    args.server
}

pub(crate) fn handle_server_cli(args: &Args) -> bool {
    if !should_handle_server_cli(args) {
        #[cfg(not(feature = "server"))]
        let _ = args.port;
        return false;
    }

    #[cfg(feature = "server")]
    {
        use rocmforge::api::server::{create_router, ModelEntry, ModelManager};

        let entry =
            ModelEntry::load(&args.model, args.draft_model.as_deref()).unwrap_or_else(|e| {
                eprintln!("Failed to load model: {}", e);
                std::process::exit(1);
            });
        let manager = ModelManager::new();
        let state = std::sync::Arc::new(manager);
        let entry_arc = std::sync::Arc::new(entry);
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
        let rt = tokio::runtime::Runtime::new()
            .expect("invariant: failed to build tokio runtime in main");
        eprintln!("rocmforge server listening on http://{}/", addr);
        rt.block_on(async {
            state
                .try_load_entry(entry_arc)
                .await
                .expect("invariant: failed to load model into server state");
            let router = create_router(state);
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .expect("invariant: failed to bind TCP address for server");
            axum::serve(listener, router)
                .await
                .expect("invariant: failed to serve HTTP router");
        });
        return true;
    }

    #[cfg(not(feature = "server"))]
    {
        eprintln!("Error: --server requires building with --features server");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{should_handle_server_cli, Args};

    fn args() -> Args {
        Args {
            model: "model.gguf".to_string(),
            prompt: "hi".to_string(),
            max_tokens: 1,
            temperature: 1.0,
            top_p: 1.0,
            no_template: false,
            list_tensors: false,
            debug: false,
            gpu: false,
            prefill_only_validate: false,
            draft_model: None,
            speculative_tokens: 0,
            kv_dump: None,
            server: false,
            port: 8080,
            threads: None,
            ctx_size: None,
            graph_map_dir: None,
            load_graph_map_dir: None,
            graph_score_metric: "neg-entropy".to_string(),
            value_head_path: None,
            rerank_top_k: 3,
            rerank_scale: 1.0,
            rerank_beam_depth: 1,
            rerank_beam_width: 1,
            rerank_beam_length_penalty: 1.0,
            train_value_head_from_traces: None,
            save_value_head: None,
            forward_graph_trace: None,
            expected_attention: None,
        }
    }

    #[test]
    fn should_handle_server_cli_tracks_server_flag() {
        let plain = args();
        assert!(!should_handle_server_cli(&plain));

        let mut server = args();
        server.server = true;
        assert!(should_handle_server_cli(&server));
    }
}
