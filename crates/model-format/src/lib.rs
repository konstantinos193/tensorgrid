//! Model format parsers for GGUF and other AI model formats.
//!
//! This crate provides parsers for reading model metadata and tensor information
//! from various model formats, starting with GGUF.

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// GGUF model parser.
pub struct GgufParser;

impl GgufParser {
    /// Parse a GGUF model file and extract metadata.
    pub fn parse<P: AsRef<Path>>(path: P) -> Result<GgufModel> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read and verify magic number
        let magic = Self::read_magic(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(anyhow::anyhow!("Invalid GGUF magic number"));
        }
        
        // Read version
        let version = reader.read_u32::<LittleEndian>()?;
        
        // Read tensor count and metadata count
        let tensor_count = reader.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()? as usize;
        
        // Read metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = Self::read_kv_pair(&mut reader)?;
            metadata.insert(key, value);
        }
        
        // Read tensor information
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(Self::read_tensor_info(&mut reader)?);
        }
        
        Ok(GgufModel {
            version,
            metadata,
            tensors,
        })
    }
    
    /// Read the GGUF magic number.
    fn read_magic<R: Read>(reader: &mut R) -> Result<u32> {
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;
        Ok(u32::from_le_bytes(magic_bytes))
    }
    
    /// Read a key-value pair from the metadata section.
    fn read_kv_pair<R: Read>(reader: &mut R) -> Result<(String, GgufValue)> {
        let key_length = reader.read_u64::<LittleEndian>()? as usize;
        let mut key_bytes = vec![0u8; key_length];
        reader.read_exact(&mut key_bytes)?;
        let key = String::from_utf8(key_bytes)?;
        
        let value_type = reader.read_u32::<LittleEndian>()?;
        let value = Self::read_value(reader, value_type)?;
        
        Ok((key, value))
    }
    
    /// Read a value based on its type.
    fn read_value<R: Read>(reader: &mut R, value_type: u32) -> Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::Uint8(reader.read_u8()?)),
            1 => Ok(GgufValue::Int8(reader.read_i8()?)),
            2 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
            3 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
            4 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
            5 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
            6 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
            7 => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
            8 => Ok(GgufValue::String(Self::read_string(reader)?)),
            9 => Ok(GgufValue::Array(Self::read_array(reader)?)),
            10 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
            11 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
            12 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
            _ => Err(anyhow::anyhow!("Unknown value type: {}", value_type)),
        }
    }
    
    /// Read a string value.
    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let length = reader.read_u64::<LittleEndian>()? as usize;
        let mut bytes = vec![0u8; length];
        reader.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    
    /// Read an array value.
    fn read_array<R: Read>(reader: &mut R) -> Result<Vec<GgufValue>> {
        let element_type = reader.read_u32::<LittleEndian>()?;
        let length = reader.read_u64::<LittleEndian>()? as usize;
        
        let mut array = Vec::with_capacity(length);
        for _ in 0..length {
            array.push(Self::read_value(reader, element_type)?);
        }
        
        Ok(array)
    }
    
    /// Read tensor information.
    fn read_tensor_info<R: Read>(reader: &mut R) -> Result<GgufTensor> {
        let name = Self::read_string(reader)?;
        let n_dims = reader.read_u32::<LittleEndian>()? as usize;
        
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(reader.read_u64::<LittleEndian>()?);
        }
        
        let type_id = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u64::<LittleEndian>()?;
        
        Ok(GgufTensor {
            name,
            shape,
            dtype: Self::dtype_from_id(type_id)?,
            offset,
        })
    }
    
    /// Convert GGUF type ID to string representation.
    fn dtype_from_id(type_id: u32) -> Result<String> {
        let dtype = match type_id {
            0 => "f32",
            1 => "f16",
            2 => "q4_0",
            3 => "q4_1",
            6 => "q5_0",
            7 => "q5_1",
            8 => "q8_0",
            9 => "q8_1",
            10 => "i8",
            11 => "i16",
            12 => "i32",
            _ => return Err(anyhow::anyhow!("Unknown type ID: {}", type_id)),
        };
        Ok(dtype.to_string())
    }
    
    /// Extract model architecture from metadata.
    pub fn get_architecture(model: &GgufModel) -> Option<String> {
        model.metadata
            .get("general.architecture")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string())
    }
      
    /// Extract model parameter count from metadata.
    pub fn get_parameter_count(model: &GgufModel) -> Option<u64> {
        model.metadata
            .get("llama.block_count")
            .and_then(|v| v.as_u64())
    }
    
    /// Extract context length from metadata.
    pub fn get_context_length(model: &GgufModel) -> Option<u64> {
        model.metadata
            .get("llama.context_length")
            .and_then(|v| v.as_u64())
    }
    
    /// Calculate total model size in bytes.
    pub fn calculate_size(model: &GgufModel) -> u64 {
        model.tensors
            .iter()
            .map(|t| {
                let elements: u64 = t.shape.iter().product();
                let bytes_per_element = dtype_bytes(&t.dtype).unwrap_or(4);
                elements * bytes_per_element
            })
            .sum()
    }
}

/// GGUF magic number.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// Parsed GGUF model.
#[derive(Debug, Clone)]
pub struct GgufModel {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensor>,
}

/// GGUF value type.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint8(v) => Some(*v as u64),
            GgufValue::Uint16(v) => Some(*v as u64),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }
    
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufValue::Int8(v) => Some(*v as i64),
            GgufValue::Int16(v) => Some(*v as i64),
            GgufValue::Int32(v) => Some(*v as i64),
            GgufValue::Int64(v) => Some(*v),
            _ => None,
        }
    }
    
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }
    
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Tensor information from GGUF.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub offset: u64,
}

/// Get bytes per element for a data type.
fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "f32" => Some(4),
        "f16" => Some(2),
        "q4_0" | "q4_1" => Some(1),
        "q5_0" | "q5_1" => Some(1),
        "q8_0" | "q8_1" => Some(1),
        "i8" => Some(1),
        "i16" => Some(2),
        "i32" => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(dtype_bytes("f32"), Some(4));
        assert_eq!(dtype_bytes("f16"), Some(2));
        assert_eq!(dtype_bytes("i8"), Some(1));
        assert_eq!(dtype_bytes("unknown"), None);
    }

    #[test]
    fn test_value_conversions() {
        let string_val = GgufValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));
        
        let uint_val = GgufValue::Uint32(42);
        assert_eq!(uint_val.as_u64(), Some(42));
        
        let bool_val = GgufValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));
    }
}
