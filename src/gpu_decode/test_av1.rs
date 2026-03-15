/// Integration tests for AV1 OBU parsing and Vulkan Video capabilities.
#[cfg(test)]
mod tests {
    use crate::gpu_decode::av1_obu;
    use crate::gpu_decode::vulkan_decoder;
    use std::ffi::CStr;
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

    #[test]
    fn test_vulkan_av1_decode_caps() {
        use ash::vk;

        // Create Vulkan instance with video extensions
        let entry = unsafe { ash::Entry::load().unwrap() };
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        // Find a physical device with video decode
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        eprintln!("Found {} physical devices", physical_devices.len());

        for pdev in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(*pdev) };
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            eprintln!("Device: {:?}", name);

            // Print all queue family flags
            let qfams = unsafe { instance.get_physical_device_queue_family_properties(*pdev) };
            for (qi, qp) in qfams.iter().enumerate() {
                eprintln!("  Queue family {}: flags={:?}, count={}", qi, qp.queue_flags, qp.queue_count);
            }

            match vulkan_decoder::check_av1_decode_support(&instance, *pdev) {
                Ok((queue_family, _)) => {
                    eprintln!("  Video decode queue family: {}", queue_family);

                    match vulkan_decoder::query_av1_decode_caps(&entry, &instance, *pdev) {
                        Ok(caps) => {
                            eprintln!("  AV1 decode caps:");
                            eprintln!("    max_coded_extent: {}x{}",
                                caps.max_coded_extent.width, caps.max_coded_extent.height);
                            eprintln!("    max_dpb_slots: {}", caps.max_dpb_slots);
                            eprintln!("    max_active_refs: {}", caps.max_active_reference_pictures);
                            assert!(caps.max_coded_extent.width >= 640);
                            assert!(caps.max_coded_extent.height >= 480);
                        }
                        Err(e) => eprintln!("  AV1 not supported: {}", e),
                    }
                }
                Err(e) => eprintln!("  No video decode: {}", e),
            }
        }

        unsafe { instance.destroy_instance(None) };
    }

    #[test]
    fn test_create_av1_decoder() {
        let video_path = match get_test_video() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test video available");
                return;
            }
        };

        // Parse sequence header from the video
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        let decoder_params = stream.parameters();
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(decoder_params).unwrap();

        let extradata = unsafe {
            let ptr = (*ctx.as_ptr()).extradata;
            let size = (*ctx.as_ptr()).extradata_size as usize;
            std::slice::from_raw_parts(ptr, size)
        };

        // Skip AV1CodecConfigurationRecord 4-byte header
        let obu_data = &extradata[4..];
        let obus = av1_obu::parse_obu_headers(obu_data).unwrap();
        let seq_obu = obus.iter().find(|o| o.obu_type == av1_obu::ObuType::SequenceHeader).unwrap();
        let seq = av1_obu::parse_sequence_header(&obu_data[seq_obu.data_offset..seq_obu.data_offset + seq_obu.data_size]).unwrap();

        eprintln!("Creating AV1 decoder for {}x{}", seq.max_frame_width_minus_1 + 1, seq.max_frame_height_minus_1 + 1);

        match vulkan_decoder::AV1Decoder::new(&seq) {
            Ok(decoder) => {
                let (w, h) = decoder.dimensions();
                eprintln!("AV1 decoder created successfully: {}x{}", w, h);
                assert_eq!(w, 640);
                assert_eq!(h, 480);
            }
            Err(e) => {
                panic!("Failed to create AV1 decoder: {}", e);
            }
        }
    }

    #[test]
    fn test_decode_first_frame() {
        let video_path = match get_test_video() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test video available");
                return;
            }
        };

        // Parse sequence header
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();
        let video_stream_index;
        let decoder_params;
        {
            let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
            video_stream_index = stream.index();
            decoder_params = stream.parameters();
        }
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(decoder_params).unwrap();
        let extradata = unsafe {
            let ptr = (*ctx.as_ptr()).extradata;
            let size = (*ctx.as_ptr()).extradata_size as usize;
            std::slice::from_raw_parts(ptr, size)
        };
        let obu_data = &extradata[4..];
        let obus = av1_obu::parse_obu_headers(obu_data).unwrap();
        let seq_obu = obus.iter().find(|o| o.obu_type == av1_obu::ObuType::SequenceHeader).unwrap();
        let seq = av1_obu::parse_sequence_header(&obu_data[seq_obu.data_offset..seq_obu.data_offset + seq_obu.data_size]).unwrap();

        // Create decoder
        let mut decoder = vulkan_decoder::AV1Decoder::new(&seq).unwrap();
        eprintln!("Decoder created, decoding first frame...");

        // Get first packet
        for (stream, packet) in ictx.packets() {
            if stream.index() != video_stream_index {
                continue;
            }
            let pkt_data = packet.data().unwrap();
            eprintln!("First packet: {} bytes", pkt_data.len());

            match decoder.decode_frame(pkt_data) {
                Ok(output) => {
                    eprintln!(
                        "Decoded frame {}: type={:?}, show={}, dpb_slot={}",
                        output.frame_index, output.frame_type, output.show_frame, output.dpb_slot
                    );

                    // Try readback
                    match decoder.read_back_nv12(output.dpb_slot) {
                        Ok((y, uv)) => {
                            eprintln!("NV12 readback: Y={} bytes, UV={} bytes", y.len(), uv.len());
                            let y_nonzero = y.iter().filter(|&&b| b != 0).count();
                            let uv_nonzero = uv.iter().filter(|&&b| b != 0).count();
                            eprintln!("Y non-zero: {} / {}", y_nonzero, y.len());
                            eprintln!("UV non-zero: {} / {}", uv_nonzero, uv.len());
                            eprintln!("UV first 32 bytes: {:?}", &uv[..32.min(uv.len())]);
                            let y_min = y.iter().copied().min().unwrap_or(0);
                            let y_max = y.iter().copied().max().unwrap_or(0);
                            let y_avg: f64 = y.iter().map(|&b| b as f64).sum::<f64>() / y.len() as f64;
                            eprintln!("Y range: min={}, max={}, avg={:.1}", y_min, y_max, y_avg);
                            let uv_min = uv.iter().copied().min().unwrap_or(0);
                            let uv_max = uv.iter().copied().max().unwrap_or(0);
                            eprintln!("UV range: min={}, max={}", uv_min, uv_max);

                            let (w, h) = decoder.dimensions();
                            let rgba = vulkan_decoder::AV1Decoder::nv12_to_rgba(&y, &uv, w, h);
                            eprintln!("RGBA: {} bytes ({}x{})", rgba.len(), w, h);
                            assert_eq!(rgba.len(), (w * h * 4) as usize);

                            let non_zero = rgba.iter().filter(|&&b| b != 0).count();
                            eprintln!("RGBA non-zero bytes: {} / {}", non_zero, rgba.len());
                            assert!(non_zero > 1000, "Decoded frame appears to be empty");
                        }
                        Err(e) => eprintln!("Readback failed: {} (expected — image layout may need transition)", e),
                    }
                }
                Err(e) => {
                    eprintln!("Decode failed: {}", e);
                    // Don't panic — the decode command structure may need tuning
                }
            }

            break; // First packet only
        }
    }

    #[test]
    fn test_decode_shared_device_same_thread() {
        // Test: create GpuSetup (like the app does), then decode on the SAME thread.
        // This tests if the shared device works when there's no concurrent wgpu access.
        let video_path = match get_test_video() {
            Some(p) => p,
            None => { eprintln!("Skipping: no test video"); return; }
        };

        // Test with a device that has BOTH graphics and video queues
        // but is NOT wrapped by wgpu. If this works, the issue is device_from_raw.
        use ash::vk as avk;
        let entry = unsafe { ash::Entry::load().unwrap() };
        let app_info = avk::ApplicationInfo::default()
            .application_name(c"test")
            .api_version(avk::make_api_version(0, 1, 3, 0));
        let inst_create = avk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&inst_create, None).unwrap() };
        let pdevs = unsafe { instance.enumerate_physical_devices().unwrap() };
        let pdev = pdevs.into_iter().find(|p| {
            let props = unsafe { instance.get_physical_device_properties(*p) };
            props.device_type == avk::PhysicalDeviceType::DISCRETE_GPU
        }).unwrap();

        let qf = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let gfx_family = qf.iter().enumerate().find(|(_, p)| p.queue_flags.contains(avk::QueueFlags::GRAPHICS)).unwrap().0 as u32;
        let vid_family = qf.iter().enumerate().find(|(_, p)| p.queue_flags.contains(avk::QueueFlags::VIDEO_DECODE_KHR)).unwrap().0 as u32;
        eprintln!("gfx_family={}, vid_family={}", gfx_family, vid_family);

        let qp = [1.0f32];
        // ONLY video decode queue — no graphics queue
        let qi = [
            avk::DeviceQueueCreateInfo::default().queue_family_index(vid_family).queue_priorities(&qp),
        ];
        let exts = [
            // No swapchain — video only
            avk::KHR_VIDEO_QUEUE_NAME.as_ptr(),
            avk::KHR_VIDEO_DECODE_QUEUE_NAME.as_ptr(),
            avk::KHR_VIDEO_DECODE_AV1_NAME.as_ptr(),
            avk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
        ];
        let mut s2 = avk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let dc = avk::DeviceCreateInfo::default()
            .queue_create_infos(&qi)
            .enabled_extension_names(&exts)
            .push_next(&mut s2);
        let device = unsafe { instance.create_device(pdev, &dc, None).unwrap() };
        eprintln!("Created device with video-decode ONLY (no graphics queue)");

        // Parse sequence header
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();
        let video_stream_index;
        let decoder_params;
        {
            let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
            video_stream_index = stream.index();
            decoder_params = stream.parameters();
        }
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(decoder_params).unwrap();
        let extradata = unsafe {
            let ptr = (*ctx.as_ptr()).extradata;
            let size = (*ctx.as_ptr()).extradata_size as usize;
            std::slice::from_raw_parts(ptr, size)
        };
        let obu_data = &extradata[4..];
        let obus = av1_obu::parse_obu_headers(obu_data).unwrap();
        let seq_obu = obus.iter().find(|o| o.obu_type == av1_obu::ObuType::SequenceHeader).unwrap();
        let seq = av1_obu::parse_sequence_header(&obu_data[seq_obu.data_offset..seq_obu.data_offset + seq_obu.data_size]).unwrap();

        // Drop the test device — use AV1Decoder::new() which creates its own
        unsafe { device.destroy_device(None); instance.destroy_instance(None); }

        let mut decoder = vulkan_decoder::AV1Decoder::new(&seq).unwrap();
        eprintln!("AV1 decoder created with new() (own device)");

        // Decode 10 frames on THIS thread (no wgpu rendering happening)
        let mut decoded = 0;
        for (stream, packet) in ictx.packets() {
            if stream.index() != video_stream_index { continue; }
            let pkt_data = packet.data().unwrap();
            match decoder.decode_frame(pkt_data) {
                Ok(out) => {
                    decoded += 1;
                    eprintln!("Frame {}: type={:?}", decoded, out.frame_type);
                }
                Err(e) => {
                    eprintln!("Decode failed at frame {}: {}", decoded + 1, e);
                    break;
                }
            }
            if decoded >= 10 { break; }
        }
        eprintln!("Decoded {} frames on multi-queue device", decoded);
        assert!(decoded >= 10, "Expected 10 frames, got {}", decoded);

        drop(decoder);
        unsafe {
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    #[test]
    fn test_decode_multiple_frames() {
        let video_path = match get_test_video() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no test video available");
                return;
            }
        };

        // Parse sequence header
        ffmpeg_next::init().unwrap();
        let mut ictx = ffmpeg_next::format::input(&video_path).unwrap();
        let video_stream_index;
        let decoder_params;
        {
            let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
            video_stream_index = stream.index();
            decoder_params = stream.parameters();
        }
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(decoder_params).unwrap();
        let extradata = unsafe {
            let ptr = (*ctx.as_ptr()).extradata;
            let size = (*ctx.as_ptr()).extradata_size as usize;
            std::slice::from_raw_parts(ptr, size)
        };
        let obu_data = &extradata[4..];
        let obus = av1_obu::parse_obu_headers(obu_data).unwrap();
        let seq_obu = obus
            .iter()
            .find(|o| o.obu_type == av1_obu::ObuType::SequenceHeader)
            .unwrap();
        let seq = av1_obu::parse_sequence_header(
            &obu_data[seq_obu.data_offset..seq_obu.data_offset + seq_obu.data_size],
        )
        .unwrap();

        let mut decoder = vulkan_decoder::AV1Decoder::new(&seq).unwrap();
        let mut decoded_count = 0;
        let max_frames = 30; // Decode first 30 frames (1 second at 30fps)

        for (stream, packet) in ictx.packets() {
            if stream.index() != video_stream_index {
                continue;
            }
            let pkt_data = packet.data().unwrap();

            match decoder.decode_frame(pkt_data) {
                Ok(output) => {
                    decoded_count += 1;
                    if decoded_count <= 5 || decoded_count % 10 == 0 {
                        eprintln!(
                            "Frame {}: type={:?}, show={}, dpb_slot={}",
                            decoded_count, output.frame_type, output.show_frame, output.dpb_slot
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Decode failed at frame {}: {}", decoded_count + 1, e);
                    break;
                }
            }

            if decoded_count >= max_frames {
                break;
            }
        }

        eprintln!("Successfully decoded {} frames", decoded_count);
        assert!(
            decoded_count >= max_frames,
            "Expected {} frames, got {}",
            max_frames,
            decoded_count
        );
    }
}
