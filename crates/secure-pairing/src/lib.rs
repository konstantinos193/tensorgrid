//! Secure node pairing and authentication.
//!
//! This crate provides secure pairing between nodes using challenge-response
//! authentication and certificate-based mutual TLS.

use cluster_types::{ClusterId, NodeId};
use anyhow::Result;
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// Pairing manager for secure node authentication.
pub struct PairingManager {
    cluster_id: ClusterId,
    pending_challenges: HashMap<String, PendingChallenge>,
    node_keys: HashMap<NodeId, NodeKeyPair>,
    coordinator_key: Keypair,
}

/// Pending pairing challenge.
#[derive(Clone)]
struct PendingChallenge {
    node_id: NodeId,
    challenge: [u8; 32],
    timestamp: chrono::DateTime<chrono::Utc>,
    expires_in_seconds: u64,
}

/// Node key pair for authentication.
#[derive(Clone, Serialize, Deserialize)]
struct NodeKeyPair {
    node_id: NodeId,
    public_key: Vec<u8>,
    certificate: DeviceCertificate,
}

/// Device certificate issued by coordinator.
#[derive(Clone, Serialize, Deserialize)]
pub struct DeviceCertificate {
    pub node_id: NodeId,
    pub cluster_id: ClusterId,
    pub public_key: Vec<u8>,
    pub issued_at: i64,
    pub expires_at: i64,
    pub coordinator_signature: Vec<u8>,
}

/// Pairing request from a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingRequest {
    pub node_name: String,
    pub hostname: String,
    pub cluster_id: ClusterId,
    pub public_key: Vec<u8>,
    pub fingerprint: String,
}

/// Pairing challenge from coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingChallenge {
    pub node_id: NodeId,
    pub challenge: Vec<u8>,
    pub coordinator_public_key: Vec<u8>,
}

/// Pairing response from node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingResponse {
    pub node_id: NodeId,
    pub challenge_response: Vec<u8>,
    pub challenge_signature: Vec<u8>,
}

impl PairingManager {
    /// Create a new pairing manager for the coordinator.
    pub fn new_coordinator(cluster_id: ClusterId) -> Result<Self> {
        let coordinator_key = Keypair::generate(&mut OsRng);
        
        Ok(Self {
            cluster_id,
            pending_challenges: HashMap::new(),
            node_keys: HashMap::new(),
            coordinator_key,
        })
    }

    /// Create a new pairing manager for a node.
    pub fn new_node(cluster_id: ClusterId) -> Result<Self> {
        let node_key = Keypair::generate(&mut OsRng);
        
        Ok(Self {
            cluster_id,
            pending_challenges: HashMap::new(),
            node_keys: HashMap::new(),
            coordinator_key: node_key,
        })
    }

    /// Generate a pairing challenge for a new node.
    pub fn generate_challenge(&mut self, request: &PairingRequest) -> Result<PairingChallenge> {
        let node_id = Uuid::new_v4();
        let mut challenge = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut challenge);

        let pending = PendingChallenge {
            node_id,
            challenge,
            timestamp: chrono::Utc::now(),
            expires_in_seconds: 300, // 5 minutes
        };

        let challenge_id = format!("{}:{}", node_id, hex::encode(&challenge[..8]));
        self.pending_challenges.insert(challenge_id, pending);

        Ok(PairingChallenge {
            node_id,
            challenge: challenge.to_vec(),
            coordinator_public_key: self.coordinator_key.public.to_bytes().to_vec(),
        })
    }

    /// Verify a pairing response from a node.
    pub fn verify_response(
        &mut self,
        response: &PairingResponse,
        node_public_key: &[u8],
    ) -> Result<DeviceCertificate> {
        let challenge_id = format!("{}:{}", response.node_id, hex::encode(&response.challenge_response[..8]));
        
        let pending = self.pending_challenges
            .remove(&challenge_id)
            .ok_or_else(|| anyhow::anyhow!("Challenge not found or expired"))?;

        // Verify challenge hasn't expired
        let elapsed = chrono::Utc::now().timestamp() - pending.timestamp.timestamp();
        if elapsed > pending.expires_in_seconds as i64 {
            return Err(anyhow::anyhow!("Challenge expired"));
        }

        // Verify the challenge response (node signed the challenge)
        let public_key = PublicKey::from_bytes(node_public_key)?;
        let signature = Signature::from_bytes(&response.challenge_signature)?;
        
        public_key.verify(&pending.challenge, &signature)?;

        // Issue device certificate
        let now = chrono::Utc::now().timestamp();
        let expires = now + (365 * 24 * 60 * 60); // 1 year

        let certificate_data = format!(
            "{}:{}:{}:{}:{}",
            response.node_id,
            self.cluster_id,
            hex::encode(node_public_key),
            now,
            expires
        );

        let mut hasher = Sha256::new();
        hasher.update(certificate_data.as_bytes());
        let hash = hasher.finalize();

        let signature = self.coordinator_key.sign(&hash);

        let certificate = DeviceCertificate {
            node_id: response.node_id,
            cluster_id: self.cluster_id,
            public_key: node_public_key.to_vec(),
            issued_at: now,
            expires_at: expires,
            coordinator_signature: signature.to_bytes().to_vec(),
        };

        // Store the node key pair
        self.node_keys.insert(
            response.node_id,
            NodeKeyPair {
                node_id: response.node_id,
                public_key: node_public_key.to_vec(),
                certificate: certificate.clone(),
            },
        );

        Ok(certificate)
    }

    /// Sign a challenge as a node.
    pub fn sign_challenge(&self, challenge: &[u8]) -> Result<Vec<u8>> {
        let signature = self.coordinator_key.sign(challenge);
        Ok(signature.to_bytes().to_vec())
    }

    /// Verify a coordinator's signature.
    pub fn verify_coordinator_signature(
        &self,
        data: &[u8],
        signature: &[u8],
        coordinator_public_key: &[u8],
    ) -> Result<bool> {
        let public_key = PublicKey::from_bytes(coordinator_public_key)?;
        let sig = Signature::from_bytes(signature)?;
        
        match public_key.verify(data, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get the public key of this instance.
    pub fn public_key(&self) -> Vec<u8> {
        self.coordinator_key.public.to_bytes().to_vec()
    }

    /// Verify a device certificate.
    pub fn verify_certificate(
        &self,
        certificate: &DeviceCertificate,
        coordinator_public_key: &[u8],
    ) -> Result<bool> {
        // Check expiration
        let now = chrono::Utc::now().timestamp();
        if now > certificate.expires_at {
            return Ok(false);
        }

        // Verify coordinator signature
        let certificate_data = format!(
            "{}:{}:{}:{}:{}",
            certificate.node_id,
            certificate.cluster_id,
            hex::encode(&certificate.public_key),
            certificate.issued_at,
            certificate.expires_at
        );

        let mut hasher = Sha256::new();
        hasher.update(certificate_data.as_bytes());
        let hash = hasher.finalize();

        self.verify_coordinator_signature(&hash, &certificate.coordinator_signature, coordinator_public_key)
    }

    /// Revoke a node's certificate.
    pub fn revoke_node(&mut self, node_id: NodeId) -> Result<()> {
        self.node_keys.remove(&node_id);
        Ok(())
    }

    /// Get all registered nodes.
    pub fn registered_nodes(&self) -> Vec<NodeId> {
        self.node_keys.keys().copied().collect()
    }

    /// Clean up expired challenges.
    pub fn cleanup_expired_challenges(&mut self) {
        let now = chrono::Utc::now();
        self.pending_challenges.retain(|_, pending| {
            let elapsed = now.timestamp() - pending.timestamp.timestamp();
            elapsed <= pending.expires_in_seconds as i64
        });
    }
}

/// Generate a device fingerprint from public key.
pub fn generate_fingerprint(public_key: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(public_key);
    let hash = hasher.finalize();
    hex::encode(&hash[..8]) // Use first 8 bytes for display
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairing_flow() {
        let cluster_id = Uuid::new_v4();
        
        // Create coordinator
        let mut coordinator = PairingManager::new_coordinator(cluster_id).unwrap();
        
        // Create node
        let node = PairingManager::new_node(cluster_id).unwrap();
        let node_public_key = node.public_key();

        // Node sends pairing request
        let request = PairingRequest {
            node_name: "test-node".to_string(),
            hostname: "test-host".to_string(),
            cluster_id,
            public_key: node_public_key.clone(),
            fingerprint: generate_fingerprint(&node_public_key),
        };

        // Coordinator generates challenge
        let challenge = coordinator.generate_challenge(&request).unwrap();
        
        // Node signs challenge
        let challenge_response = challenge.challenge.clone();
        let challenge_signature = node.sign_challenge(&challenge_response).unwrap();

        let response = PairingResponse {
            node_id: challenge.node_id,
            challenge_response,
            challenge_signature,
        };

        // Coordinator verifies response
        let certificate = coordinator.verify_response(&response, &node_public_key).unwrap();
        
        assert_eq!(certificate.node_id, challenge.node_id);
        assert_eq!(certificate.cluster_id, cluster_id);
    }

    #[test]
    fn test_certificate_verification() {
        let cluster_id = Uuid::new_v4();
        let mut coordinator = PairingManager::new_coordinator(cluster_id).unwrap();
        let coordinator_public_key = coordinator.public_key();

        let node = PairingManager::new_node(cluster_id).unwrap();
        let node_public_key = node.public_key();

        let request = PairingRequest {
            node_name: "test-node".to_string(),
            hostname: "test-host".to_string(),
            cluster_id,
            public_key: node_public_key.clone(),
            fingerprint: generate_fingerprint(&node_public_key),
        };

        let challenge = coordinator.generate_challenge(&request).unwrap();
        let challenge_response = challenge.challenge.clone();
        let challenge_signature = node.sign_challenge(&challenge_response).unwrap();

        let response = PairingResponse {
            node_id: challenge.node_id,
            challenge_response,
            challenge_signature,
        };

        let certificate = coordinator.verify_response(&response, &node_public_key).unwrap();
        
        // Verify the certificate
        let valid = coordinator.verify_certificate(&certificate, &coordinator_public_key).unwrap();
        assert!(valid);
    }
}