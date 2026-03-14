/// Integration test: extract AV1 OBUs from a real video file using ffmpeg,
/// then parse the sequence header and frame headers with our parser.
#[cfg(test)]
mod tests {
    use crate::gpu_decode::av1_obu;
    use std::path::Path;

    fn get_test_video() -> Option<std::path::PathBuf> {
        let path = dirs::home_dir()?
            .join(".cache/huggingface/lerobot/gtgando/so101_pick_place_smolvla/videos/chunk-000/observation.images.wrist/episode_000000.mp4");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn test_parse_real_av1_sequence_header() {
        let video_path = match get_test_video() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test video available");
                return;
            }
        };

        // Use ffmpeg to extract the first few packets
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();

        let video_stream_index;
        let decoder_params;
        {
            let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
            video_stream_index = stream.index();
            decoder_params = stream.parameters();
        }

        // The codec extradata contains the AV1 configration record
        // which includes the sequence header OBU.
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(decoder_params).unwrap();
        let extradata = unsafe {
            let ptr = (*ctx.as_ptr()).extradata;
            let size = (*ctx.as_ptr()).extradata_size as usize;
            if ptr.is_null() || size == 0 {
                eprintln!("No extradata in codec context");
                return;
            }
            std::slice::from_raw_parts(ptr, size)
        };

        eprintln!("Extradata size: {} bytes", extradata.len());
        eprintln!("Extradata hex: {:02x?}", &extradata[..extradata.len().min(32)]);

        // AV1CodecConfigurationRecord starts with a 4-byte config header,
        // followed by OBUs (configOBUs). Skip the 4-byte header.
        if extradata.len() > 4 {
            let obu_data = &extradata[4..];
            match av1_obu::parse_obu_headers(obu_data) {
                Ok(obus) => {
                    eprintln!("Found {} OBUs in extradata", obus.len());
                    for obu in &obus {
                        eprintln!("  OBU: {:?}, offset={}, data_offset={}, data_size={}",
                            obu.obu_type, obu.offset, obu.data_offset, obu.data_size);

                        if obu.obu_type == av1_obu::ObuType::SequenceHeader {
                            let payload = &obu_data[obu.data_offset..obu.data_offset + obu.data_size];
                            match av1_obu::parse_sequence_header(payload) {
                                Ok(seq) => {
                                    eprintln!("Sequence Header parsed successfully!");
                                    eprintln!("  profile: {}", seq.seq_profile);
                                    eprintln!("  max_frame_size: {}x{}",
                                        seq.max_frame_width_minus_1 + 1,
                                        seq.max_frame_height_minus_1 + 1);
                                    eprintln!("  bit_depth: {}", seq.color_config.bit_depth);
                                    eprintln!("  enable_order_hint: {}", seq.enable_order_hint);
                                    eprintln!("  order_hint_bits: {}", seq.order_hint_bits);
                                    eprintln!("  enable_cdef: {}", seq.enable_cdef);
                                    eprintln!("  enable_restoration: {}", seq.enable_restoration);
                                    eprintln!("  film_grain: {}", seq.film_grain_params_present);

                                    assert_eq!(seq.max_frame_width_minus_1 + 1, 640);
                                    assert_eq!(seq.max_frame_height_minus_1 + 1, 480);
                                    assert_eq!(seq.color_config.bit_depth, 8);
                                }
                                Err(e) => panic!("Failed to parse sequence header: {}", e),
                            }
                        }
                    }
                }
                Err(e) => panic!("Failed to parse OBU headers: {}", e),
            }
        }

        // Also try parsing a packet's OBUs (frame data)
        for (stream, packet) in ictx.packets() {
            if stream.index() != video_stream_index {
                continue;
            }
            let pkt_data = packet.data().unwrap();
            eprintln!("\nFirst packet: {} bytes", pkt_data.len());

            match av1_obu::parse_obu_headers(pkt_data) {
                Ok(obus) => {
                    eprintln!("Found {} OBUs in packet", obus.len());
                    for obu in &obus {
                        eprintln!("  OBU: {:?}, data_size={}", obu.obu_type, obu.data_size);
                    }
                }
                Err(e) => eprintln!("Failed to parse packet OBUs: {}", e),
            }
            break; // Only first packet
        }
    }
}
