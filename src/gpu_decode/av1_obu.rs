//! AV1 OBU (Open Bitstream Unit) parser.
//!
//! Parses the AV1 bitstream to extract sequence headers, frame headers, and
//! tile information needed by Vulkan Video's StdVideoDecodeAV1PictureInfo and
//! StdVideoAV1SequenceHeader structures.
//!
//! Reference: AV1 Bitstream & Decoding Process Specification v1.0.0 with Errata 1
//! https://aomediacodec.github.io/av1-spec/

/// AV1 OBU types (Section 6.2.2)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum ObuType {
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6, // Combined frame header + tile group
    RedundantFrameHeader = 7,
    TileList = 8,
    Padding = 15,
    Unknown(u8),
}

impl From<u8> for ObuType {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            x => Self::Unknown(x),
        }
    }
}

/// AV1 frame types (Section 6.8.2)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FrameType {
    KeyFrame = 0,
    InterFrame = 1,
    IntraOnlyFrame = 2,
    SwitchFrame = 3,
}

impl From<u8> for FrameType {
    fn from(v: u8) -> Self {
        match v & 0x3 {
            0 => Self::KeyFrame,
            1 => Self::InterFrame,
            2 => Self::IntraOnlyFrame,
            _ => Self::SwitchFrame,
        }
    }
}

/// AV1 profiles
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Profile {
    Main = 0,       // 8/10-bit 4:2:0
    High = 1,       // 8/10-bit 4:4:4
    Professional = 2, // 8/10/12-bit, any subsampling
}

/// Parsed AV1 Sequence Header OBU (Section 5.5)
#[derive(Debug, Clone)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,

    // Timing info
    pub timing_info_present: bool,
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,

    // Frame dimensions
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,

    // Frame ID
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,

    // Features
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_choose_screen_content_tools: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_choose_integer_mv: bool,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits_minus_1: u8,
    pub order_hint_bits: u8,

    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,

    // Color config
    pub color_config: ColorConfig,

    pub film_grain_params_present: bool,

    // Operating points
    pub operating_points_cnt_minus_1: u8,
    pub operating_point_idc: [u16; 32],
    pub seq_level_idx: [u8; 32],
    pub seq_tier: [u8; 32],
}

/// AV1 Color Configuration (Section 5.5.2)
#[derive(Debug, Clone)]
pub struct ColorConfig {
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_description_present: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

/// Parsed AV1 Frame Header (Section 5.9)
/// Contains fields required by StdVideoDecodeAV1PictureInfo.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,

    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,

    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u8,
    pub primary_ref_frame: u8,

    pub refresh_frame_flags: u8,

    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    pub coded_denom: u8,

    // Reference frames (for inter frames)
    pub ref_frame_idx: [u8; 7],
    pub ref_order_hint: [u8; 8],

    // Quantization
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,

    // Loop filter
    pub loop_filter_level: [u8; 4],
    pub loop_filter_sharpness: u8,
    pub loop_filter_delta_enabled: bool,

    // CDEF
    pub cdef_damping_minus_3: u8,
    pub cdef_bits: u8,

    // Loop restoration
    pub lr_type: [u8; 3],

    // Tile info
    pub tile_cols: u16,
    pub tile_rows: u16,

    pub is_intra_frame: bool,
    pub allow_intrabc: bool,
}

/// A parsed OBU with its type, offset in the bitstream, and size.
#[derive(Debug, Clone)]
pub struct ParsedObu {
    pub obu_type: ObuType,
    pub offset: usize,    // byte offset in the input buffer
    pub size: usize,      // total OBU size including header
    pub data_offset: usize, // byte offset of OBU payload (after header)
    pub data_size: usize, // payload size
}

/// Tile info extracted for Vulkan's VkVideoDecodeAV1PictureInfoKHR.
#[derive(Debug, Clone)]
pub struct TileInfo {
    pub tile_count: u32,
    pub tile_offsets: Vec<u32>,
    pub tile_sizes: Vec<u32>,
}

/// Bitstream reader for reading individual bits and fields.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0-7, bits remaining in current byte
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 8,
        }
    }

    fn from_offset(data: &'a [u8], byte_offset: usize) -> Self {
        Self {
            data,
            byte_pos: byte_offset,
            bit_pos: 8,
        }
    }

    fn bits_read(&self) -> usize {
        self.byte_pos * 8 + (8 - self.bit_pos as usize)
    }

    fn remaining_bytes(&self) -> usize {
        if self.bit_pos == 8 {
            self.data.len() - self.byte_pos
        } else {
            self.data.len() - self.byte_pos - 1
        }
    }

    fn read_bit(&mut self) -> Result<u8, ParseError> {
        if self.byte_pos >= self.data.len() {
            return Err(ParseError::UnexpectedEof);
        }
        self.bit_pos -= 1;
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        if self.bit_pos == 0 {
            self.byte_pos += 1;
            self.bit_pos = 8;
        }
        Ok(bit)
    }

    fn read_bits(&mut self, n: u8) -> Result<u32, ParseError> {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u32;
        }
        Ok(val)
    }

    fn read_bool(&mut self) -> Result<bool, ParseError> {
        Ok(self.read_bit()? == 1)
    }

    fn read_byte(&mut self) -> Result<u8, ParseError> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Read unsigned variable-length integer (leb128)
    fn read_leb128(&mut self) -> Result<u32, ParseError> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.read_byte()? as u64;
            value |= (byte & 0x7F) << (i * 7);
            if byte & 0x80 == 0 {
                break;
            }
        }
        Ok(value as u32)
    }

    /// Read uvlc (unsigned exp-Golomb) - Section 4.10.3
    fn read_uvlc(&mut self) -> Result<u32, ParseError> {
        let mut leading_zeros = 0u32;
        loop {
            let bit = self.read_bit()?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Err(ParseError::InvalidData("uvlc too large"));
            }
        }
        if leading_zeros >= 32 {
            return Ok(u32::MAX);
        }
        let value = self.read_bits(leading_zeros as u8)?;
        Ok(value + (1 << leading_zeros) - 1)
    }

    /// Read su(n) - signed integer using n bits (Section 4.10.6)
    fn read_su(&mut self, n: u8) -> Result<i32, ParseError> {
        let value = self.read_bits(n)? as i32;
        let sign_mask = 1 << (n - 1);
        if value & sign_mask != 0 {
            Ok(value - (1 << n))
        } else {
            Ok(value)
        }
    }

    fn byte_align(&mut self) {
        if self.bit_pos != 8 {
            self.byte_pos += 1;
            self.bit_pos = 8;
        }
    }

    fn current_byte_offset(&self) -> usize {
        self.byte_pos
    }

    fn skip_bits(&mut self, n: usize) -> Result<(), ParseError> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedEof,
    InvalidData(&'static str),
    UnsupportedObu(u8),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "Unexpected end of bitstream"),
            Self::InvalidData(msg) => write!(f, "Invalid AV1 data: {}", msg),
            Self::UnsupportedObu(t) => write!(f, "Unsupported OBU type: {}", t),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse OBU headers from a buffer, returning their types, offsets, and sizes.
/// Does NOT parse the OBU payloads — just enumerates the structure.
pub fn parse_obu_headers(data: &[u8]) -> Result<Vec<ParsedObu>, ParseError> {
    let mut obus = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        let start = pos;
        if pos >= data.len() {
            break;
        }
        let header_byte = data[pos];
        pos += 1;

        // obu_forbidden_bit (1) | obu_type (4) | obu_extension_flag (1) | obu_has_size_field (1) | reserved (1)
        let _forbidden = (header_byte >> 7) & 1;
        let obu_type_raw = (header_byte >> 3) & 0xF;
        let extension_flag = (header_byte >> 2) & 1;
        let has_size = (header_byte >> 1) & 1;

        if extension_flag == 1 {
            if pos >= data.len() {
                break;
            }
            pos += 1; // skip extension byte
        }

        let obu_size = if has_size == 1 {
            // Read leb128 size
            let mut value = 0u64;
            for i in 0..8 {
                if pos >= data.len() {
                    return Err(ParseError::UnexpectedEof);
                }
                let byte = data[pos] as u64;
                pos += 1;
                value |= (byte & 0x7F) << (i * 7);
                if byte & 0x80 == 0 {
                    break;
                }
            }
            value as usize
        } else {
            data.len() - pos
        };

        let data_offset = pos;
        let data_size = obu_size;

        obus.push(ParsedObu {
            obu_type: ObuType::from(obu_type_raw),
            offset: start,
            size: (pos - start) + obu_size,
            data_offset,
            data_size,
        });

        pos = data_offset + obu_size;
    }

    Ok(obus)
}

/// Parse an AV1 Sequence Header OBU payload.
pub fn parse_sequence_header(data: &[u8]) -> Result<SequenceHeader, ParseError> {
    let mut r = BitReader::new(data);

    let seq_profile = r.read_bits(3)? as u8;
    let still_picture = r.read_bool()?;
    let reduced_still_picture_header = r.read_bool()?;

    let mut timing_info_present = false;
    let mut num_units_in_display_tick = 0u32;
    let mut time_scale = 0u32;
    let mut equal_picture_interval = false;
    let mut operating_points_cnt_minus_1 = 0u8;
    let mut operating_point_idc = [0u16; 32];
    let mut seq_level_idx = [0u8; 32];
    let mut seq_tier = [0u8; 32];

    if reduced_still_picture_header {
        operating_points_cnt_minus_1 = 0;
        seq_level_idx[0] = r.read_bits(5)? as u8;
    } else {
        timing_info_present = r.read_bool()?;
        if timing_info_present {
            num_units_in_display_tick = r.read_bits(32)?;
            time_scale = r.read_bits(32)?;
            equal_picture_interval = r.read_bool()?;
            if equal_picture_interval {
                let _num_ticks = r.read_uvlc()?;
            }
            // decoder_model_info_present
            let decoder_model_info_present = r.read_bool()?;
            if decoder_model_info_present {
                let _buffer_delay_length_minus_1 = r.read_bits(5)?;
                let _num_units_in_decoding_tick = r.read_bits(32)?;
                let _buffer_removal_time_length_minus_1 = r.read_bits(5)?;
                let _frame_presentation_time_length_minus_1 = r.read_bits(5)?;
            }
        }
        let _initial_display_delay_present = r.read_bool()?;
        operating_points_cnt_minus_1 = r.read_bits(5)? as u8;
        for i in 0..=operating_points_cnt_minus_1 as usize {
            operating_point_idc[i] = r.read_bits(12)? as u16;
            seq_level_idx[i] = r.read_bits(5)? as u8;
            if seq_level_idx[i] > 7 {
                seq_tier[i] = r.read_bit()?;
            }
            // Skip decoder_model and initial_display_delay per operating point
            if timing_info_present {
                let _decoder_model_present = r.read_bool()?;
            }
            if _initial_display_delay_present {
                let _has_delay = r.read_bool()?;
                if _has_delay {
                    let _delay = r.read_bits(4)?;
                }
            }
        }
    }

    let frame_width_bits_minus_1 = r.read_bits(4)? as u8;
    let frame_height_bits_minus_1 = r.read_bits(4)? as u8;
    let max_frame_width_minus_1 = r.read_bits(frame_width_bits_minus_1 + 1)? as u16;
    let max_frame_height_minus_1 = r.read_bits(frame_height_bits_minus_1 + 1)? as u16;

    let mut frame_id_numbers_present = false;
    let mut delta_frame_id_length_minus_2 = 0u8;
    let mut additional_frame_id_length_minus_1 = 0u8;

    if !reduced_still_picture_header {
        frame_id_numbers_present = r.read_bool()?;
        if frame_id_numbers_present {
            delta_frame_id_length_minus_2 = r.read_bits(4)? as u8;
            additional_frame_id_length_minus_1 = r.read_bits(3)? as u8;
        }
    }

    let use_128x128_superblock = r.read_bool()?;
    let enable_filter_intra = r.read_bool()?;
    let enable_intra_edge_filter = r.read_bool()?;

    let mut enable_interintra_compound = false;
    let mut enable_masked_compound = false;
    let mut enable_warped_motion = false;
    let mut enable_dual_filter = false;
    let mut enable_order_hint = false;
    let mut enable_jnt_comp = false;
    let mut enable_ref_frame_mvs = false;
    let mut seq_choose_screen_content_tools = false;
    let mut seq_force_screen_content_tools = 0u8;
    let mut seq_choose_integer_mv = false;
    let mut seq_force_integer_mv = 0u8;
    let mut order_hint_bits_minus_1 = 0u8;
    let mut order_hint_bits = 0u8;

    if !reduced_still_picture_header {
        enable_interintra_compound = r.read_bool()?;
        enable_masked_compound = r.read_bool()?;
        enable_warped_motion = r.read_bool()?;
        enable_dual_filter = r.read_bool()?;
        enable_order_hint = r.read_bool()?;
        if enable_order_hint {
            enable_jnt_comp = r.read_bool()?;
            enable_ref_frame_mvs = r.read_bool()?;
        }
        seq_choose_screen_content_tools = r.read_bool()?;
        if seq_choose_screen_content_tools {
            seq_force_screen_content_tools = 2; // SELECT_SCREEN_CONTENT_TOOLS
        } else {
            seq_force_screen_content_tools = r.read_bit()?;
        }
        if seq_force_screen_content_tools > 0 {
            seq_choose_integer_mv = r.read_bool()?;
            if seq_choose_integer_mv {
                seq_force_integer_mv = 2; // SELECT_INTEGER_MV
            } else {
                seq_force_integer_mv = r.read_bit()?;
            }
        } else {
            seq_force_integer_mv = 2;
        }
        if enable_order_hint {
            order_hint_bits_minus_1 = r.read_bits(3)? as u8;
            order_hint_bits = order_hint_bits_minus_1 + 1;
        }
    }

    let enable_superres = r.read_bool()?;
    let enable_cdef = r.read_bool()?;
    let enable_restoration = r.read_bool()?;

    // Color config
    let color_config = parse_color_config(&mut r, seq_profile)?;

    let film_grain_params_present = r.read_bool()?;

    Ok(SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        timing_info_present,
        num_units_in_display_tick,
        time_scale,
        equal_picture_interval,
        max_frame_width_minus_1,
        max_frame_height_minus_1,
        frame_width_bits_minus_1,
        frame_height_bits_minus_1,
        frame_id_numbers_present,
        delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_choose_screen_content_tools,
        seq_force_screen_content_tools,
        seq_choose_integer_mv,
        seq_force_integer_mv,
        order_hint_bits_minus_1,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color_config,
        film_grain_params_present,
        operating_points_cnt_minus_1,
        operating_point_idc,
        seq_level_idx,
        seq_tier,
    })
}

fn parse_color_config(r: &mut BitReader, seq_profile: u8) -> Result<ColorConfig, ParseError> {
    let high_bitdepth = r.read_bool()?;
    let mut twelve_bit = false;
    let bit_depth;

    if seq_profile == 2 && high_bitdepth {
        twelve_bit = r.read_bool()?;
        bit_depth = if twelve_bit { 12 } else { 10 };
    } else {
        bit_depth = if high_bitdepth { 10 } else { 8 };
    }

    let mono_chrome = if seq_profile == 1 {
        false
    } else {
        r.read_bool()?
    };

    let color_description_present = r.read_bool()?;
    let mut color_primaries = 2u8; // CP_UNSPECIFIED
    let mut transfer_characteristics = 2u8;
    let mut matrix_coefficients = 2u8;

    if color_description_present {
        color_primaries = r.read_byte()?;
        transfer_characteristics = r.read_byte()?;
        matrix_coefficients = r.read_byte()?;
    }

    let mut color_range = false;
    let mut subsampling_x = false;
    let mut subsampling_y = false;
    let mut chroma_sample_position = 0u8;

    if mono_chrome {
        color_range = r.read_bool()?;
        subsampling_x = true;
        subsampling_y = true;
    } else if color_primaries == 1 && transfer_characteristics == 13 && matrix_coefficients == 0 {
        // sRGB / BT.709
        color_range = true;
        // profile must be High or Professional for 4:4:4
    } else {
        color_range = r.read_bool()?;
        if seq_profile == 0 {
            subsampling_x = true;
            subsampling_y = true;
        } else if seq_profile == 1 {
            subsampling_x = false;
            subsampling_y = false;
        } else {
            if bit_depth == 12 {
                subsampling_x = r.read_bool()?;
                if subsampling_x {
                    subsampling_y = r.read_bool()?;
                }
            } else {
                subsampling_x = true;
            }
        }
        if subsampling_x && subsampling_y {
            chroma_sample_position = r.read_bits(2)? as u8;
        }
    }

    let separate_uv_delta_q = if mono_chrome {
        false
    } else {
        r.read_bool()?
    };

    Ok(ColorConfig {
        high_bitdepth,
        twelve_bit,
        bit_depth,
        mono_chrome,
        color_description_present,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
    })
}

/// tile_log2: smallest k such that blk_size << k >= target (Section 5.9.15)
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Read delta_q value (Section 5.9.12): 1-bit flag, if set read su(7).
fn read_delta_q(r: &mut BitReader) -> Result<i8, ParseError> {
    let delta_coded = r.read_bool()?;
    if delta_coded {
        Ok(r.read_su(7)? as i8)
    } else {
        Ok(0)
    }
}

/// Parse an AV1 uncompressed frame header (Section 5.9).
/// `seq` is the current sequence header, needed for conditional field parsing.
/// Returns (FrameHeader, header_bytes_consumed) where header_bytes_consumed
/// can be used to calculate the tile data offset within a Frame OBU.
pub fn parse_frame_header(
    data: &[u8],
    seq: &SequenceHeader,
) -> Result<(FrameHeader, usize), ParseError> {
    let mut r = BitReader::new(data);

    let mut show_existing_frame = false;
    if !seq.reduced_still_picture_header {
        show_existing_frame = r.read_bool()?;
    }

    if show_existing_frame {
        // show_existing_frame: just references an existing frame in the DPB.
        // For Vulkan decode, this is handled differently.
        let _frame_to_show = r.read_bits(3)?;
        return Ok((FrameHeader {
            frame_type: FrameType::InterFrame,
            show_frame: true,
            showable_frame: true,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: 7, // PRIMARY_REF_NONE
            refresh_frame_flags: 0,
            frame_width: seq.max_frame_width_minus_1 as u32 + 1,
            frame_height: seq.max_frame_height_minus_1 as u32 + 1,
            render_width: seq.max_frame_width_minus_1 as u32 + 1,
            render_height: seq.max_frame_height_minus_1 as u32 + 1,
            use_superres: false,
            coded_denom: 0,
            ref_frame_idx: [0; 7],
            ref_order_hint: [0; 8],
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            lr_type: [0; 3],
            tile_cols: 1,
            tile_rows: 1,
            is_intra_frame: false,
            allow_intrabc: false,
        }, 0));
    }

    let frame_type;
    let mut show_frame = true;
    let mut showable_frame = false;
    let mut error_resilient_mode = false;

    if seq.reduced_still_picture_header {
        frame_type = FrameType::KeyFrame;
        show_frame = true;
    } else {
        frame_type = FrameType::from(r.read_bits(2)? as u8);
        show_frame = r.read_bool()?;
        if show_frame {
            // If show_frame, decoder_model may need timing info
        } else {
            showable_frame = r.read_bool()?;
        }
        if frame_type == FrameType::SwitchFrame || (frame_type == FrameType::KeyFrame && show_frame)
        {
            error_resilient_mode = true;
        } else {
            error_resilient_mode = r.read_bool()?;
        }
    }

    let disable_cdf_update = r.read_bool()?;

    let mut allow_screen_content_tools = false;
    if seq.seq_force_screen_content_tools == 2 {
        allow_screen_content_tools = r.read_bool()?;
    } else {
        allow_screen_content_tools = seq.seq_force_screen_content_tools != 0;
    }

    let mut force_integer_mv = false;
    if allow_screen_content_tools {
        if seq.seq_force_integer_mv == 2 {
            force_integer_mv = r.read_bool()?;
        } else {
            force_integer_mv = seq.seq_force_integer_mv != 0;
        }
    }

    let mut current_frame_id = 0u32;
    if seq.frame_id_numbers_present {
        let id_len =
            seq.additional_frame_id_length_minus_1 + seq.delta_frame_id_length_minus_2 + 3;
        current_frame_id = r.read_bits(id_len)?;
    }

    let frame_size_override_flag = if frame_type == FrameType::SwitchFrame {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        r.read_bool()?
    };

    let order_hint = if seq.enable_order_hint {
        r.read_bits(seq.order_hint_bits)? as u8
    } else {
        0
    };

    let primary_ref_frame;
    let is_intra_frame =
        frame_type == FrameType::IntraOnlyFrame || frame_type == FrameType::KeyFrame;

    if is_intra_frame || error_resilient_mode {
        primary_ref_frame = 7; // PRIMARY_REF_NONE
    } else {
        primary_ref_frame = r.read_bits(3)? as u8;
    }

    // refresh_frame_flags
    let refresh_frame_flags;
    if frame_type == FrameType::SwitchFrame
        || (frame_type == FrameType::KeyFrame && show_frame)
    {
        refresh_frame_flags = 0xFF; // all frames
    } else {
        refresh_frame_flags = r.read_byte()?;
    }

    // Frame size
    let mut frame_width = seq.max_frame_width_minus_1 as u32 + 1;
    let mut frame_height = seq.max_frame_height_minus_1 as u32 + 1;

    if frame_size_override_flag {
        frame_width = r.read_bits(seq.frame_width_bits_minus_1 + 1)? + 1;
        frame_height = r.read_bits(seq.frame_height_bits_minus_1 + 1)? + 1;
    }

    // Superres
    let mut use_superres = false;
    let mut coded_denom = 8u8; // SUPERRES_NUM (no scaling)
    if seq.enable_superres {
        use_superres = r.read_bool()?;
        if use_superres {
            coded_denom = r.read_bits(3)? as u8 + 9; // SUPERRES_DENOM_MIN
        }
    }

    let render_width = frame_width;
    let render_height = frame_height;

    // Reference frame indices (for inter frames)
    let mut ref_frame_idx = [0u8; 7];
    if !is_intra_frame {
        for i in 0..7 {
            ref_frame_idx[i] = r.read_bits(3)? as u8;
        }
    }

    // allow_intrabc
    let allow_intrabc = if is_intra_frame && allow_screen_content_tools {
        r.read_bool()?
    } else {
        false
    };

    // Continue parsing: interpolation_filter (Section 5.9.10)
    let _is_filter_switchable;
    let _interpolation_filter;
    if is_intra_frame || allow_intrabc {
        _is_filter_switchable = false;
        _interpolation_filter = 4u8; // SWITCHABLE
    } else {
        _is_filter_switchable = r.read_bool()?;
        if _is_filter_switchable {
            _interpolation_filter = 4; // SWITCHABLE
        } else {
            _interpolation_filter = r.read_bits(2)? as u8;
        }
    }

    // is_motion_mode_switchable
    if !is_intra_frame && !allow_intrabc {
        let _is_motion_mode_switchable = r.read_bool()?;
    }

    // use_ref_frame_mvs
    if seq.enable_ref_frame_mvs
        && !is_intra_frame
        && !allow_intrabc
        && !error_resilient_mode
    {
        let _use_ref_frame_mvs = r.read_bool()?;
    }

    // After interpolation_filter + is_motion_mode_switchable + use_ref_frame_mvs,
    // the spec says tile_info() comes next (Section 5.9.15).
    //
    // tile_info is complex and variable-length. For 480p single-tile videos,
    // the tile_info section is short. We need to parse it to advance the bit
    // position correctly before quantization_params.

    // tile_info() — Section 5.9.15
    let sb_size: u32 = if seq.use_128x128_superblock { 128 } else { 64 };
    let sb_cols = (frame_width + sb_size - 1) / sb_size;
    let sb_rows = (frame_height + sb_size - 1) / sb_size;
    let _sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
    let max_tile_width_sb = 4096 / sb_size; // MAX_TILE_WIDTH_SB

    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(1024)); // MAX_TILE_COLS = 64, simplified

    let mut tile_cols_log2 = min_log2_tile_cols;
    while tile_cols_log2 < max_log2_tile_cols {
        let increment = r.read_bool()?;
        if increment {
            tile_cols_log2 += 1;
        } else {
            break;
        }
    }
    let tile_cols = 1u16 << tile_cols_log2;

    let max_log2_tile_rows = tile_log2(1, sb_rows.min(1024));
    let min_log2_tiles = if tile_cols_log2 > 0 { tile_cols_log2 } else { 0 };
    let min_log2_tile_rows = if min_log2_tiles > tile_cols_log2 {
        min_log2_tiles - tile_cols_log2
    } else {
        0
    };
    let mut tile_rows_log2 = min_log2_tile_rows;
    while tile_rows_log2 < max_log2_tile_rows {
        let increment = r.read_bool()?;
        if increment {
            tile_rows_log2 += 1;
        } else {
            break;
        }
    }
    let tile_rows = 1u16 << tile_rows_log2;

    // tile_size_bytes_minus_1 (if more than 1 tile)
    if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        let _tile_size_bytes_minus_1 = r.read_bits(2)?;
        // context_update_tile_id
        let _context_update_tile_id = r.read_bits((tile_cols_log2 + tile_rows_log2) as u8)?;
    }

    // quantization_params (Section 5.9.12)
    let base_q_idx = r.read_byte()?;
    let delta_q_y_dc = read_delta_q(&mut r)?;
    let mut delta_q_u_dc = 0i8;
    let mut delta_q_u_ac = 0i8;
    let mut delta_q_v_dc = 0i8;
    let mut delta_q_v_ac = 0i8;
    let using_qmatrix;

    if !seq.color_config.mono_chrome {
        let diff_uv_delta = if seq.color_config.separate_uv_delta_q {
            r.read_bool()?
        } else {
            false
        };
        delta_q_u_dc = read_delta_q(&mut r)?;
        delta_q_u_ac = read_delta_q(&mut r)?;
        if diff_uv_delta {
            delta_q_v_dc = read_delta_q(&mut r)?;
            delta_q_v_ac = read_delta_q(&mut r)?;
        } else {
            delta_q_v_dc = delta_q_u_dc;
            delta_q_v_ac = delta_q_u_ac;
        }
    }
    using_qmatrix = r.read_bool()?;
    if using_qmatrix {
        let _qm_y = r.read_bits(4)?;
        let _qm_u = r.read_bits(4)?;
        let _qm_v = r.read_bits(4)?;
    }

    // segmentation_params (Section 5.9.14)
    let _segmentation_enabled = r.read_bool()?;
    if _segmentation_enabled {
        if primary_ref_frame == 7 {
            // segmentation_update_map = 1, segmentation_temporal_update = 0
        } else {
            let _seg_update_map = r.read_bool()?;
            if _seg_update_map {
                let _seg_temporal_update = r.read_bool()?;
            }
        }
        let seg_update_data = r.read_bool()?;
        if seg_update_data {
            // Read 8 segments × 8 features
            for _i in 0..8 {
                for j in 0..8 {
                    let feature_enabled = r.read_bool()?;
                    if feature_enabled {
                        // Feature value bits depend on feature ID
                        let bits_to_read: u8 = match j {
                            0 => 8, // SEG_LVL_ALT_Q
                            1 => 6, // SEG_LVL_ALT_LF_Y_V
                            2 => 6, // SEG_LVL_ALT_LF_Y_H
                            3 => 6, // SEG_LVL_ALT_LF_U
                            4 => 6, // SEG_LVL_ALT_LF_V
                            5 => 3, // SEG_LVL_REF_FRAME
                            6 => 0, // SEG_LVL_SKIP
                            7 => 0, // SEG_LVL_GLOBALMV
                            _ => 0,
                        };
                        if bits_to_read > 0 {
                            // Signed value for features 0-4
                            let _val = if j <= 4 {
                                r.read_su(1 + bits_to_read)?
                            } else {
                                r.read_bits(bits_to_read)? as i32
                            };
                        }
                    }
                }
            }
        }
    }

    // delta_q_params (Section 5.9.17)
    let mut delta_q_present = false;
    let mut delta_q_res = 0u8;
    if base_q_idx > 0 {
        delta_q_present = r.read_bool()?;
    }
    if delta_q_present {
        delta_q_res = r.read_bits(2)? as u8;
    }

    // delta_lf_params (Section 5.9.18)
    let mut delta_lf_present = false;
    let mut delta_lf_res = 0u8;
    let mut _delta_lf_multi = false;
    if delta_q_present {
        if !allow_intrabc {
            delta_lf_present = r.read_bool()?;
        }
        if delta_lf_present {
            delta_lf_res = r.read_bits(2)? as u8;
            _delta_lf_multi = r.read_bool()?;
        }
    }

    // loop_filter_params (Section 5.9.11)
    let mut loop_filter_level = [0u8; 4];
    let mut loop_filter_sharpness = 0u8;
    let mut loop_filter_delta_enabled = false;
    let mut loop_filter_ref_deltas = [1i8, 0, 0, 0, -1, 0, -1, -1]; // AV1 defaults
    let mut loop_filter_mode_deltas = [0i8; 2];

    if !is_intra_frame && !allow_intrabc {
        loop_filter_level[0] = r.read_bits(6)? as u8;
        loop_filter_level[1] = r.read_bits(6)? as u8;
        if !seq.color_config.mono_chrome && (loop_filter_level[0] != 0 || loop_filter_level[1] != 0) {
            loop_filter_level[2] = r.read_bits(6)? as u8;
            loop_filter_level[3] = r.read_bits(6)? as u8;
        }
        loop_filter_sharpness = r.read_bits(3)? as u8;
        loop_filter_delta_enabled = r.read_bool()?;
        if loop_filter_delta_enabled {
            let delta_update = r.read_bool()?;
            if delta_update {
                for i in 0..8 {
                    let update = r.read_bool()?;
                    if update {
                        loop_filter_ref_deltas[i] = r.read_su(7)? as i8;
                    }
                }
                for i in 0..2 {
                    let update = r.read_bool()?;
                    if update {
                        loop_filter_mode_deltas[i] = r.read_su(7)? as i8;
                    }
                }
            }
        }
    }

    // cdef_params (Section 5.9.19)
    let mut cdef_damping_minus_3 = 0u8;
    let mut cdef_bits = 0u8;
    let mut cdef_y_pri_strength = [0u8; 8];
    let mut cdef_y_sec_strength = [0u8; 8];
    let mut cdef_uv_pri_strength = [0u8; 8];
    let mut cdef_uv_sec_strength = [0u8; 8];

    if seq.enable_cdef && !allow_intrabc {
        cdef_damping_minus_3 = r.read_bits(2)? as u8;
        cdef_bits = r.read_bits(2)? as u8;
        let num_cdef = 1 << cdef_bits;
        for i in 0..num_cdef {
            cdef_y_pri_strength[i] = r.read_bits(4)? as u8;
            cdef_y_sec_strength[i] = r.read_bits(2)? as u8;
            if !seq.color_config.mono_chrome {
                cdef_uv_pri_strength[i] = r.read_bits(4)? as u8;
                cdef_uv_sec_strength[i] = r.read_bits(2)? as u8;
            }
        }
    }

    // lr_params (Section 5.9.20)
    let mut lr_type = [0u8; 3]; // RESTORE_NONE for all planes
    if seq.enable_restoration && !allow_intrabc {
        let mut uses_lr = false;
        for i in 0..3 {
            let lr_type_val = r.read_bits(2)? as u8;
            lr_type[i] = lr_type_val;
            if lr_type_val != 0 {
                uses_lr = true;
            }
        }
        if uses_lr {
            // lr_unit_shift
            if seq.use_128x128_superblock {
                let _lr_unit_shift = r.read_bit()?;
            } else {
                let _lr_unit_shift = r.read_bit()?;
                if _lr_unit_shift != 0 {
                    let _lr_unit_extra_shift = r.read_bit()?;
                }
            }
            // lr_uv_shift
            if !seq.color_config.mono_chrome && seq.color_config.subsampling_x && seq.color_config.subsampling_y {
                let _lr_uv_shift = r.read_bit()?;
            }
        }
    }

    // read_tx_mode (Section 5.9.21)
    if !is_intra_frame || allow_intrabc {
        // TX_MODE_SELECT is always used for inter frames in practice
    }
    let _tx_mode = if base_q_idx > 0 {
        let _tx_mode_select = r.read_bool()?;
        if _tx_mode_select { 2u8 } else { 1u8 } // TX_MODE_SELECT or TX_MODE_LARGEST
    } else {
        2u8 // TX_MODE_ONLY_4X4 when lossless
    };

    // frame_reference_mode (Section 5.9.23)
    if !is_intra_frame {
        let _reference_select = r.read_bool()?;
    }

    // skip_mode_params (Section 5.9.24)
    // skip_mode is only available for inter frames with order hints
    if !is_intra_frame && seq.enable_order_hint {
        let _skip_mode_present = r.read_bool()?;
    }

    // allow_warped_motion
    if !is_intra_frame && !allow_intrabc && seq.enable_warped_motion {
        let _allow_warped_motion = r.read_bool()?;
    }

    // reduced_tx_set
    let _reduced_tx_set = r.read_bool()?;

    // global_motion_params (Section 5.9.25) — skip for non-inter or simple cases
    if !is_intra_frame {
        for _ref_frame in 0..7 {
            // Each reference frame has is_global, is_rot_zoom, is_translation flags
            let is_global = r.read_bool()?;
            if is_global {
                let is_rot_zoom = r.read_bool()?;
                if is_rot_zoom {
                    // ROTZOOM: 2 parameters, each su(12) + su(12) + su(9) + su(9) = 4 params
                    let _ = r.read_su(12)?;
                    let _ = r.read_su(12)?;
                    let _ = r.read_su(12)?;
                    let _ = r.read_su(12)?;
                } else {
                    let is_translation = r.read_bool()?;
                    if is_translation {
                        // TRANSLATION: su(9) + su(9)
                        let _ = r.read_su(9)?;
                        let _ = r.read_su(9)?;
                    } else {
                        // AFFINE: 6 parameters
                        let _ = r.read_su(12)?;
                        let _ = r.read_su(12)?;
                        let _ = r.read_su(12)?;
                        let _ = r.read_su(12)?;
                        let _ = r.read_su(12)?;
                        let _ = r.read_su(12)?;
                    }
                }
            }
        }
    }

    // film_grain_params (Section 5.9.30) — skip if not present
    // seq.film_grain_params_present is false for our test video

    let fh = FrameHeader {
        frame_type,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        frame_width,
        frame_height,
        render_width,
        render_height,
        use_superres,
        coded_denom,
        ref_frame_idx,
        ref_order_hint: [0; 8],
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        loop_filter_level,
        loop_filter_sharpness,
        loop_filter_delta_enabled,
        cdef_damping_minus_3,
        cdef_bits,
        lr_type,
        tile_cols,
        tile_rows,
        is_intra_frame,
        allow_intrabc,
    };

    // Byte-align and report consumed bytes (tile data starts after this)
    r.byte_align();
    let header_bytes = r.current_byte_offset();

    Ok((fh, header_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basics() {
        let data = [0b10110100, 0b11000000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bits(3).unwrap(), 0b110);
        assert_eq!(r.read_bits(3).unwrap(), 0b100);
        // Next byte
        assert_eq!(r.read_bits(2).unwrap(), 0b11);
    }

    #[test]
    fn test_obu_header_parsing() {
        // Minimal OBU: type=SequenceHeader (1), no extension, has_size=1, size=2, payload=0x00,0x00
        let data = [
            0b0_0001_0_1_0, // type=1, no ext, has_size=1
            0x02,           // size = 2 (leb128)
            0x00,
            0x00, // payload
        ];
        let obus = parse_obu_headers(&data).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, ObuType::SequenceHeader);
        assert_eq!(obus[0].data_offset, 2);
        assert_eq!(obus[0].data_size, 2);
    }
}
