//! On-chain redemption of winning positions (EOA custody).
//!
//! With `risk.auto_redeem = true` (magic/pro accounts) Polymarket redeems the
//! winning tokens automatically and the orphan checker only books the
//! REDEEM in the DB. With `auto_redeem = false` (normal account, the ERC1155s live
//! in the EOA of `POLY_PRIVATE_KEY`) nobody cashes anything in: the money stays in the
//! wallet until `redeemPositions` of the Conditional Token Framework is called.
//! This module implements that path over the official SDK's CTF client.
//!
//! Safety order (fail-closed): the on-chain redeem is attempted BEFORE
//! `apply_settlement`. Since `list_expired_unresolved` requires
//! `outcome_won IS NULL`, any failure leaves the position pending and the
//! orphan's next poll retries it — it is never marked redeemed without a tx.
//!
//! `balanceOf` pre-check: if the EOA no longer holds the token (redeemed
//! in an earlier run between the send and the settlement, or redeemed by hand),
//! another tx is NOT sent — a second `redeemPositions` on a zero balance
//! reverts, burning gas every cycle. If the balance is in the proxy/funder instead
//! of the EOA, custody is not with this account: it is left pending with a
//! warn (manual redemption, never mark `redeemed=1`).

use std::str::FromStr;
use std::sync::Arc;

use alloy::network::EthereumWallet;
use alloy::primitives::{address, Address, B256, U256};
use alloy::providers::{Provider, ProviderBuilder};
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use anyhow::Context as _;
use polymarket_client_sdk_v2::ctf::types::{RedeemNegRiskRequest, RedeemPositionsRequest};
use tracing::{info, warn};

use crate::types::Position;

/// Polygon mainnet CTF (same address as the SDK's `contract_config`).
const CTF_MAINNET: Address = address!("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045");
/// USDC.e on Polygon — collateral of all Polymarket markets.
const USDC_E: Address = address!("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174");
const POLYGON_CHAIN_ID: u64 = 137;

sol! {
    #[sol(rpc)]
    interface IErc1155Balance {
        function balanceOf(address account, uint256 id) external view returns (uint256);
    }
}

/// Result of an on-chain redemption attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedeemOutcome {
    /// Tx sent and mined — holds the hash for `positions.redeem_tx`.
    Sent(String),
    /// The EOA no longer holds the token: redeemed earlier (crash between the send
    /// and the settlement) or by hand. Booked with mark `onchain-no-balance`.
    NoBalance,
    /// The balance is in the proxy/funder, not the EOA — this account cannot
    /// redeem it; manual redemption required. Left pending.
    ProxyCustody,
}

/// Injectable port: the real implementation talks to Polygon; tests
/// fake it to fix the behavior of `settlement::orphan::settle` without a chain.
pub trait ChainRedeemer: Send + Sync {
    fn redeem<'a>(
        &'a self,
        pos: &'a Position,
        token_index: usize,
        outcome_slots: usize,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<RedeemOutcome>> + Send + 'a>,
    >;
}

/// Index sets of a binary YES/NO market: `[1, 2]` redeems the winning side
/// whichever it is (CTF interprets each index set as an outcome to cash in).
pub fn binary_index_sets() -> Vec<U256> {
    vec![U256::from(1u64), U256::from(2u64)]
}

/// `amounts` for `NegRiskAdapter.redeemPositions`: the full balance in the
/// slot of our token, zero elsewhere. The token's index within
/// `market.tokens` matches the adapter's outcome slot (CLOB order).
pub fn amounts_neg_risk(token_index: usize, outcome_slots: usize, balance: U256) -> Vec<U256> {
    let mut v = vec![U256::ZERO; outcome_slots.max(1)];
    if token_index < v.len() {
        v[token_index] = balance;
    }
    v
}

/// Real redeemer over the SDK's CTF client + signing HTTP provider.
/// `DynProvider` avoids naming the builder's tower of fillers.
pub struct CtfRedeemer {
    ctf: polymarket_client_sdk_v2::ctf::Client<alloy::providers::DynProvider>,
    erc1155: IErc1155Balance::IErc1155BalanceInstance<alloy::providers::DynProvider>,
    eoa: Address,
    funder: Option<Address>,
    collateral: Address,
}

impl CtfRedeemer {
    /// `rpc_url` = Polygon HTTPS endpoint; `signer` = EOA that holds the
    /// tokens; `funder` = proxy wallet if the account has one (`None` = plain
    /// EOA). Empty `collateral` → mainnet USDC.e.
    pub fn new(
        rpc_url: &str,
        signer: Arc<PrivateKeySigner>,
        funder: Option<Address>,
        collateral: &str,
    ) -> anyhow::Result<Self> {
        let url: reqwest::Url = rpc_url
            .parse()
            .with_context(|| format!("redeem.rpc_url invalid: {rpc_url:?}"))?;
        let eoa = signer.address();
        let wallet = EthereumWallet::from(signer.as_ref().clone());
        let provider = ProviderBuilder::new()
            .wallet(wallet)
            .connect_http(url)
            .erased();
        let ctf = polymarket_client_sdk_v2::ctf::Client::with_neg_risk(
            provider.clone(),
            POLYGON_CHAIN_ID,
        )
        .map_err(|e| anyhow::anyhow!("CTF client init: {e}"))?;
        let erc1155 = IErc1155Balance::new(CTF_MAINNET, provider);
        let collateral = if collateral.is_empty() {
            USDC_E
        } else {
            Address::from_str(collateral)
                .with_context(|| format!("redeem.collateral invalid: {collateral:?}"))?
        };
        Ok(Self {
            ctf,
            erc1155,
            eoa,
            funder,
            collateral,
        })
    }

    async fn balance_of(&self, owner: Address, token_id: U256) -> anyhow::Result<U256> {
        self.erc1155
            .balanceOf(owner, token_id)
            .call()
            .await
            .map_err(|e| anyhow::anyhow!("balanceOf({owner}) failed: {e}"))
    }

    async fn redeem_inner(
        &self,
        pos: &Position,
        token_index: usize,
        outcome_slots: usize,
    ) -> anyhow::Result<RedeemOutcome> {
        let token_u256 = U256::from_str(&pos.token_id)
            .with_context(|| format!("token_id not numeric: {:?}", pos.token_id))?;
        let eoa_balance = self.balance_of(self.eoa, token_u256).await?;

        if eoa_balance.is_zero() {
            // The EOA holds nothing: either it was already redeemed or the balance lives in the
            // proxy. Tell them apart before blindly marking `redeemed=1`.
            if let Some(funder) = self.funder {
                if funder != self.eoa {
                    let funder_balance = self.balance_of(funder, token_u256).await?;
                    if !funder_balance.is_zero() {
                        return Ok(RedeemOutcome::ProxyCustody);
                    }
                }
            }
            return Ok(RedeemOutcome::NoBalance);
        }

        let condition_id = B256::from_str(&pos.market_cid)
            .with_context(|| format!("market_cid is not bytes32: {:?}", pos.market_cid))?;

        let hash = if pos.neg_risk {
            let amounts = amounts_neg_risk(token_index, outcome_slots, eoa_balance);
            let req = RedeemNegRiskRequest::builder()
                .condition_id(condition_id)
                .amounts(amounts)
                .build();
            self.ctf
                .redeem_neg_risk(&req)
                .await
                .map_err(|e| anyhow::anyhow!("redeem_neg_risk failed: {e}"))?
                .transaction_hash
        } else {
            let req = RedeemPositionsRequest::builder()
                .collateral_token(self.collateral)
                .condition_id(condition_id)
                .index_sets(binary_index_sets())
                .build();
            self.ctf
                .redeem_positions(&req)
                .await
                .map_err(|e| anyhow::anyhow!("redeem_positions failed: {e}"))?
                .transaction_hash
        };

        info!(
            token = %pos.token_id,
            cid   = %pos.market_cid,
            tx    = %hash,
            balance = %eoa_balance,
            "redeem: redeemPositions mined"
        );
        Ok(RedeemOutcome::Sent(format!("{hash}")))
    }
}

impl ChainRedeemer for CtfRedeemer {
    fn redeem<'a>(
        &'a self,
        pos: &'a Position,
        token_index: usize,
        outcome_slots: usize,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<RedeemOutcome>> + Send + 'a>,
    > {
        Box::pin(self.redeem_inner(pos, token_index, outcome_slots))
    }
}

/// Decides the `redeem_tx` for `settlement::orphan::settle` in live without auto-redeem.
/// Returns `Err` when the position must stay pending (tx failure or
/// proxy custody) — the next poll retries it.
pub async fn try_redeem(
    redeemer: Option<&dyn ChainRedeemer>,
    pos: &Position,
    won: bool,
    token_index: usize,
    outcome_slots: usize,
) -> anyhow::Result<Option<String>> {
    // Only winners hold a redeemable balance; losers are settled
    // anyway and stay redeemed=0 (nothing to claim on-chain).
    let Some(r) = redeemer else { return Ok(None) };
    if !won {
        return Ok(None);
    }
    match r.redeem(pos, token_index, outcome_slots).await? {
        RedeemOutcome::Sent(hash) => Ok(Some(hash)),
        RedeemOutcome::NoBalance => {
            warn!(
                token = %pos.token_id,
                cid   = %pos.market_cid,
                "redeem: on-chain balance 0 — already redeemed; booked with forensic mark"
            );
            Ok(Some("onchain-no-balance".to_string()))
        }
        RedeemOutcome::ProxyCustody => {
            anyhow::bail!(
                "redeem: the token {} balance is in the proxy/funder, not the EOA — \
                 manual redemption pending; the position will be retried",
                pos.token_id
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_index_sets_are_1_and_2() {
        assert_eq!(
            binary_index_sets(),
            vec![U256::from(1u64), U256::from(2u64)]
        );
    }

    #[test]
    fn amounts_neg_risk_places_balance_in_its_slot() {
        let v = amounts_neg_risk(1, 2, U256::from(7u64));
        assert_eq!(v, vec![U256::ZERO, U256::from(7u64)]);
    }

    #[test]
    fn amounts_neg_risk_with_out_of_range_index_does_not_panic() {
        let v = amounts_neg_risk(9, 2, U256::from(7u64));
        assert_eq!(v, vec![U256::ZERO, U256::ZERO]);
    }
}
