#![allow(clippy::arithmetic_side_effects)]
use {
    clap::value_t,
    log::*,
    solana_bench_tps::{
        bench::{do_bench_tps, max_lamports_for_prioritization},
        bench_tps_client::BenchTpsClient,
        cli::{self, ExternalClientType},
        keypairs::get_keypairs,
        send_batch::{generate_durable_nonce_accounts, generate_keypairs},
    },
    solana_client::{
        connection_cache::ConnectionCache,
        thin_client::ThinClient,
        tpu_client::{TpuClient, TpuClientConfig},
    },
    solana_genesis::Base64Account,
    solana_gossip::gossip_service::{discover_cluster, get_client, get_multi_client},
    solana_rpc_client::rpc_client::RpcClient,
    solana_sdk::{
        commitment_config::CommitmentConfig,
        fee_calculator::FeeRateGovernor,
        pubkey::Pubkey,
        signature::{Keypair, Signer},
        system_program,
    },
    solana_streamer::{socket::SocketAddrSpace, streamer::StakedNodes},
    std::{
        collections::HashMap,
        fs::File,
        io::prelude::*,
        net::{IpAddr, SocketAddr},
        path::Path,
        process::exit,
        sync::{Arc, RwLock},
        thread::sleep,
        time::Duration,
        fs,
    },
};

/// Number of signatures for all transactions in ~1 week at ~100K TPS
pub const NUM_SIGNATURES_FOR_TXS: u64 = 100_000 * 60 * 60 * 24 * 7;

/// Request information about node's stake
/// If fail to get requested information, return error
/// Otherwise return stake of the node
/// along with total activated stake of the network
fn find_node_activated_stake(
    rpc_client: Arc<RpcClient>,
    node_id: Pubkey,
) -> Result<(u64, u64), ()> {
    let vote_accounts = rpc_client.get_vote_accounts();
    if let Err(error) = vote_accounts {
        error!("Failed to get vote accounts, error: {}", error);
        return Err(());
    }

    let vote_accounts = vote_accounts.unwrap();

    let total_active_stake: u64 = vote_accounts
        .current
        .iter()
        .map(|vote_account| vote_account.activated_stake)
        .sum();

    let node_id_as_str = node_id.to_string();
    let find_result = vote_accounts
        .current
        .iter()
        .find(|&vote_account| vote_account.node_pubkey == node_id_as_str);
    match find_result {
        Some(value) => Ok((value.activated_stake, total_active_stake)),
        None => {
            error!("Failed to find stake for requested node");
            Err(())
        }
    }
}

fn create_connection_cache(
    json_rpc_url: &str,
    tpu_connection_pool_size: usize,
    use_quic: bool,
    bind_address: IpAddr,
    client_node_id: Option<&Keypair>,
) -> ConnectionCache {
    if !use_quic {
        return ConnectionCache::with_udp(
            "bench-tps-connection_cache_udp",
            tpu_connection_pool_size,
        );
    }
    if client_node_id.is_none() {
        return ConnectionCache::new_quic(
            "bench-tps-connection_cache_quic",
            tpu_connection_pool_size,
        );
    }

    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        json_rpc_url.to_string(),
        CommitmentConfig::confirmed(),
    ));

    let client_node_id = client_node_id.unwrap();
    let (stake, total_stake) =
        find_node_activated_stake(rpc_client, client_node_id.pubkey()).unwrap_or_default();
    info!("Stake for specified client_node_id: {stake}, total stake: {total_stake}");
    let stakes = HashMap::from([
        (client_node_id.pubkey(), stake),
        (Pubkey::new_unique(), total_stake - stake),
    ]);
    let staked_nodes = Arc::new(RwLock::new(StakedNodes::new(
        Arc::new(stakes),
        HashMap::<Pubkey, u64>::default(), // overrides
    )));
    ConnectionCache::new_with_client_options(
        "bench-tps-connection_cache_quic",
        tpu_connection_pool_size,
        None,
        Some((client_node_id, bind_address)),
        Some((&staked_nodes, &client_node_id.pubkey())),
    )
}

#[allow(clippy::too_many_arguments)]
fn create_client(
    external_client_type: &ExternalClientType,
    entrypoint_addr: &SocketAddr,
    json_rpc_url: &str,
    websocket_url: &str,
    multi_client: bool,
    rpc_tpu_sockets: Option<(SocketAddr, SocketAddr)>,
    num_nodes: usize,
    target_node: Option<Pubkey>,
    connection_cache: ConnectionCache,
) -> Arc<dyn BenchTpsClient + Send + Sync> {
    match external_client_type {
        ExternalClientType::RpcClient => Arc::new(RpcClient::new_with_commitment(
            json_rpc_url.to_string(),
            CommitmentConfig::confirmed(),
        )),
        ExternalClientType::ThinClient => {
            let connection_cache = Arc::new(connection_cache);
            if let Some((rpc, tpu)) = rpc_tpu_sockets {
                Arc::new(ThinClient::new(rpc, tpu, connection_cache))
            } else {
                let nodes =
                    discover_cluster(entrypoint_addr, num_nodes, SocketAddrSpace::Unspecified)
                        .unwrap_or_else(|err| {
                            eprintln!("Failed to discover {num_nodes} nodes: {err:?}");
                            exit(1);
                        });
                if multi_client {
                    let (client, num_clients) =
                        get_multi_client(&nodes, &SocketAddrSpace::Unspecified, connection_cache);
                    if nodes.len() < num_clients {
                        eprintln!(
                            "Error: Insufficient nodes discovered.  Expecting {num_nodes} or more"
                        );
                        exit(1);
                    }
                    Arc::new(client)
                } else if let Some(target_node) = target_node {
                    info!("Searching for target_node: {:?}", target_node);
                    let mut target_client = None;
                    for node in nodes {
                        if node.pubkey() == &target_node {
                            target_client = Some(get_client(
                                &[node],
                                &SocketAddrSpace::Unspecified,
                                connection_cache,
                            ));
                            break;
                        }
                    }
                    Arc::new(target_client.unwrap_or_else(|| {
                        eprintln!("Target node {target_node} not found");
                        exit(1);
                    }))
                } else {
                    Arc::new(get_client(
                        &nodes,
                        &SocketAddrSpace::Unspecified,
                        connection_cache,
                    ))
                }
            }
        }
        ExternalClientType::TpuClient => {
            let rpc_client = Arc::new(RpcClient::new_with_commitment(
                json_rpc_url.to_string(),
                CommitmentConfig::confirmed(),
            ));
            match connection_cache {
                ConnectionCache::Udp(cache) => Arc::new(
                    TpuClient::new_with_connection_cache(
                        rpc_client,
                        websocket_url,
                        TpuClientConfig::default(),
                        cache,
                    )
                    .unwrap_or_else(|err| {
                        eprintln!("Could not create TpuClient {err:?}");
                        exit(1);
                    }),
                ),
                ConnectionCache::Quic(cache) => Arc::new(
                    TpuClient::new_with_connection_cache(
                        rpc_client,
                        websocket_url,
                        TpuClientConfig::default(),
                        cache,
                    )
                    .unwrap_or_else(|err| {
                        eprintln!("Could not create TpuClient {err:?}");
                        exit(1);
                    }),
                ),
            }
        }
    }
}

fn main() {
    solana_logger::setup_with_default("solana=info");
    solana_metrics::set_panic_hook("bench-tps", /*version:*/ None);

    let matches = cli::build_args(solana_version::version!()).get_matches();
    let cli_config = match cli::parse_args(&matches) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            exit(1);
        }
    };

    let cli::Config {
        entrypoint_addr,
        json_rpc_url,
        websocket_url,
        id,
        num_nodes,
        tx_count,
        keypair_multiplier,
        client_ids_and_stake_file,
        write_to_client_file,
        read_from_client_file,
        target_lamports_per_signature,
        multi_client,
        num_lamports_per_account,
        target_node,
        external_client_type,
        use_quic,
        tpu_connection_pool_size,
        compute_unit_price,
        use_durable_nonce,
        instruction_padding_config,
        bind_address,
        client_node_id,
        parallel_bench_clients,
        ..
    } = &cli_config;

    let keypair_count = *tx_count * keypair_multiplier;
    if *write_to_client_file {
        info!("Generating {} keypairs", keypair_count);
        let (keypairs, _) = generate_keypairs(id, keypair_count as u64);
        let num_accounts = keypairs.len() as u64;
        let max_fee = FeeRateGovernor::new(*target_lamports_per_signature, 0)
            .max_lamports_per_signature
            .saturating_add(max_lamports_for_prioritization(compute_unit_price));
        let num_lamports_per_account = (num_accounts - 1 + NUM_SIGNATURES_FOR_TXS * max_fee)
            / num_accounts
            + num_lamports_per_account;
        let mut accounts = HashMap::new();
        keypairs.iter().for_each(|keypair| {
            accounts.insert(
                serde_json::to_string(&keypair.to_bytes().to_vec()).unwrap(),
                Base64Account {
                    balance: num_lamports_per_account,
                    executable: false,
                    owner: system_program::id().to_string(),
                    data: String::new(),
                },
            );
        });

        info!("Writing {}", client_ids_and_stake_file);
        let serialized = serde_yaml::to_string(&accounts).unwrap();
        let path = Path::new(&client_ids_and_stake_file);
        let mut file = File::create(path).unwrap();
        file.write_all(b"---\n").unwrap();
        file.write_all(&serialized.into_bytes()).unwrap();
        return;
    }

    info!("Connecting to the cluster");
    let rpc_tpu_sockets: Option<(SocketAddr, SocketAddr)> =
        if let Ok(rpc_addr) = value_t!(matches, "rpc_addr", String) {
            let rpc = rpc_addr.parse().unwrap_or_else(|e| {
                eprintln!("RPC address should parse as socketaddr {e:?}");
                exit(1);
            });
            let tpu = value_t!(matches, "tpu_addr", String)
                .unwrap()
                .parse()
                .unwrap_or_else(|e| {
                    eprintln!("TPU address should parse to a socket: {e:?}");
                    exit(1);
                });
            Some((rpc, tpu))
        } else {
            None
        };

    let connection_cache = create_connection_cache(
        json_rpc_url,
        *tpu_connection_pool_size,
        *use_quic,
        *bind_address,
        client_node_id.as_ref(),
    );
    let client = create_client(
        external_client_type,
        entrypoint_addr,
        json_rpc_url,
        websocket_url,
        *multi_client,
        rpc_tpu_sockets,
        *num_nodes,
        *target_node,
        connection_cache,
    );
    if let Some(instruction_padding_config) = instruction_padding_config {
        info!(
            "Checking for existence of instruction padding program: {}",
            instruction_padding_config.program_id
        );
        client
            .get_account(&instruction_padding_config.program_id)
            .expect("Instruction padding program must be deployed to this cluster. Deploy the program using `solana program deploy ./bench-tps/tests/fixtures/spl_instruction_padding.so` and pass the resulting program id with `--instruction-padding-program-id`");
    }
    let keypairs = get_keypairs(
        client.clone(),
        id,
        keypair_count,
        *num_lamports_per_account,
        client_ids_and_stake_file,
        *read_from_client_file,
        instruction_padding_config.is_some(),
    );

    let nonce_keypairs = if *use_durable_nonce {
        Some(generate_durable_nonce_accounts(client.clone(), &keypairs))
    } else {
        None
    };

    check_all_parallel_clients(*parallel_bench_clients, id.pubkey());
    info!("Ready to start the test!!!");

    do_bench_tps(client, cli_config, keypairs, nonce_keypairs);
}

const WATCH_DIR: &str = "./";
const WATCH_FILE_PREFIX: &str = "ready_";
const WATCH_CHECK_PERIOD_US: u64 = 1000;

fn check_all_parallel_clients(parallel_bench_clients: usize, key: Pubkey) {
    info!("number of parallel clients {parallel_bench_clients}");
    let file_path = format!("{}{}", WATCH_FILE_PREFIX, key.to_string());
    let _ = File::create(file_path);
    let path = Path::new(WATCH_DIR);
    let target_file_count = parallel_bench_clients;
    loop {
        let current_count = count_matching_files(&path);
        if current_count == target_file_count {
            break;
        }
        sleep(Duration::from_micros(WATCH_CHECK_PERIOD_US));
    }
}

fn count_matching_files(dir: &Path) -> usize {
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok()) // Filter out entries with errors
        .filter(|entry| entry.file_name().to_str().unwrap().starts_with(WATCH_FILE_PREFIX)) // Check if filename starts with "abc_"
        .count()
}
