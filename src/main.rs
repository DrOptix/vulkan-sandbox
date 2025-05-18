use std::{
    alloc::{Layout, alloc},
    borrow::Cow,
    ffi::{CStr, CString, c_char},
    ptr, slice,
};

use anyhow::{Context, Result};
use ash::{
    ext::debug_utils,
    khr,
    vk::{self, PhysicalDevice},
};
use glfw::{Glfw, GlfwReceiver};

fn main() {
    match HelloTriangleApplication::new(800, 600, "Hello triangle") {
        Ok(mut app) => {
            app.run();
        }
        Err(err) => {
            eprint!("Error: {err}");
        }
    };
}

#[derive(Debug, Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[derive(Debug, Default)]
struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct HelloTriangleApplication {
    glfw: Glfw,
    window: glfw::PWindow,
    window_events: GlfwReceiver<(f64, glfw::WindowEvent)>,
    window_surface: vk::SurfaceKHR,
    _entry: ash::Entry,
    instance: ash::Instance,
    khr_surface_instance: khr::surface::Instance,
    khr_swapchain_device: khr::swapchain::Device,
    debug_utils_instance: Option<debug_utils::Instance>,
    debug_utils_messanger: Option<vk::DebugUtilsMessengerEXT>,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    swapchain_images_views: Vec<vk::ImageView>,
    _swapchain_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
}

/// Public functions
impl HelloTriangleApplication {
    pub fn new(width: u32, height: u32, title: &str) -> Result<Self> {
        let mut glfw = glfw::init_no_callbacks()?;
        let (window, window_events) = Self::create_window(&mut glfw, width, height, title)?;

        let mut extensions = glfw
            .get_required_instance_extensions()
            .context("Vulkan is not supported")?;

        let mut validation_layers = vec![];
        if cfg!(debug_assertions) {
            extensions.push("VK_EXT_debug_utils".to_string());
            validation_layers.push("VK_LAYER_KHRONOS_validation".to_string());
        }

        let entry = unsafe { ash::Entry::load() }?;
        let instance = Self::create_instance(&entry, &extensions, &validation_layers)?;
        let khr_surface_instance = khr::surface::Instance::new(&entry, &instance);
        let maybe_debug_messanger = Self::setup_debug_messenger(&entry, &instance)?;
        let debug_utils_instance = maybe_debug_messanger.as_ref().map(|debug| debug.0.clone());
        let debug_utils_messanger = maybe_debug_messanger.as_ref().map(|debug| debug.1);
        let window_surface = Self::create_window_surface(&instance, &window)?;
        let physical_device =
            Self::pick_physical_device(&instance, &khr_surface_instance, window_surface)?;
        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &instance,
            &khr_surface_instance,
            physical_device,
            window_surface,
        )?;
        let khr_swapchain_device = khr::swapchain::Device::new(&instance, &device);
        let (
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        ) = Self::create_swap_chain(
            &window,
            &instance,
            &khr_surface_instance,
            &khr_swapchain_device,
            physical_device,
            &device,
            window_surface,
        )?;

        let render_pass = Self::create_render_pass(&device, swapchain_format)?;
        let pipeline_layout = Self::create_graphics_pipeline(&device)?;

        Ok(Self {
            glfw,
            window,
            window_events,
            window_surface,
            _entry: entry,
            instance,
            khr_surface_instance,
            khr_swapchain_device,
            debug_utils_instance,
            debug_utils_messanger,
            _physical_device: physical_device,
            device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            swapchain,
            _swapchain_images: swapchain_images,
            swapchain_images_views: swapchain_image_views,
            _swapchain_format: swapchain_format,
            _swapchain_extent: swapchain_extent,
            render_pass,
            pipeline_layout,
        })
    }

    pub fn run(&mut self) {
        self.window.set_key_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&self.window_events) {
                println!("{event:?}");
                if let glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) = event
                {
                    self.window.set_should_close(true)
                }
            }
        }
    }
}

/// Internal functions
impl HelloTriangleApplication {
    fn create_window(
        glfw: &mut Glfw,
        width: u32,
        height: u32,
        title: &str,
    ) -> Result<(glfw::PWindow, GlfwReceiver<(f64, glfw::WindowEvent)>)> {
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(false));

        if let Some(window_and_events) =
            glfw.create_window(width, height, title, glfw::WindowMode::Windowed)
        {
            Ok(window_and_events)
        } else {
            anyhow::bail!("Unable to create GLFW window")
        }
    }

    fn create_window_surface(
        instance: &ash::Instance,
        window: &glfw::PWindow,
    ) -> Result<vk::SurfaceKHR> {
        let mut surface: std::mem::MaybeUninit<vk::SurfaceKHR> = std::mem::MaybeUninit::uninit();

        if window.create_window_surface(instance.handle(), ptr::null(), surface.as_mut_ptr())
            != vk::Result::SUCCESS
        {
            anyhow::bail!("Failed to create GLFW Vulkan window surface.");
        }

        Ok(unsafe { surface.assume_init() })
    }

    fn create_instance(
        entry: &ash::Entry,
        extensions: &[String],
        validation_layers: &[String],
    ) -> Result<ash::Instance> {
        if cfg!(debug_assertions) {
            Self::check_validation_layers_support(entry, validation_layers)?;
        }

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3)
            .engine_name(c"hello_trangle")
            .engine_version(0)
            .application_name(c"hello_trangle")
            .engine_version(0);

        let extensions: Vec<CString> = extensions
            .iter()
            .map(|ext| CString::new(ext.as_str()).unwrap())
            .collect();

        let extensions: Vec<*const c_char> = extensions
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();

        let validation_layers: Vec<CString> = validation_layers
            .iter()
            .map(|layer| CString::new(layer.as_str()).unwrap())
            .collect();

        let validation_layers: Vec<*const c_char> = validation_layers
            .iter()
            .map(|layer| layer.as_ptr() as *const c_char)
            .collect();

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&validation_layers)
            .application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(instance)
    }

    fn check_validation_layers_support(
        entry: &ash::Entry,
        requested_layers: &[String],
    ) -> Result<()> {
        let existing_layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .context("Failed to enumerate instance layer properties")?
        };

        for layer in requested_layers {
            let mut found = false;
            for prop in &existing_layer_properties {
                let layer_name = unsafe {
                    CStr::from_ptr(prop.layer_name.as_ptr())
                        .to_str()
                        .context("Failed to convert layer name to string")?
                };

                if layer_name.eq(layer.as_str()) {
                    found = true;
                    break;
                }
            }

            if !found {
                anyhow::bail!("Validation layer '{}' not found", layer);
            }
        }

        Ok(())
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<bool> {
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };

        for extension in available_extensions.into_iter() {
            let extension_name = unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr())
                    .to_str()
                    .context("Failed to convert extension name to string")?
            };

            if extension_name.eq("VK_KHR_swapchain") {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
        if cfg!(debug_assertions) {
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(Self::vulkan_debug_callback));

            let debug_utils_instance = debug_utils::Instance::new(entry, instance);
            let debug_utils_messanger =
                unsafe { debug_utils_instance.create_debug_utils_messenger(&create_info, None)? };

            return Ok(Some((debug_utils_instance, debug_utils_messanger)));
        }

        Ok(None)
    }

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        unsafe {
            let callback_data = *p_callback_data;
            let message_id_number = callback_data.message_id_number;

            let message_id_name = if callback_data.p_message_id_name.is_null() {
                Cow::from("")
            } else {
                CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
            };

            let message = if callback_data.p_message.is_null() {
                Cow::from("")
            } else {
                CStr::from_ptr(callback_data.p_message).to_string_lossy()
            };

            println!(
                "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
            );

            vk::FALSE
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            anyhow::bail!("Failed to find devices with Vulkan support");
        }

        for physical_device in physical_devices.into_iter() {
            if Self::is_device_suitable(instance, khr_surface_instance, physical_device, surface)? {
                return Ok(physical_device);
            }
        }

        anyhow::bail!("Unable to find a suitable device");
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<bool> {
        let queue_family_indices =
            Self::find_queue_families(instance, khr_surface_instance, physical_device, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, physical_device)?;

        let mut swap_chain_supported = false;
        if extensions_supported {
            let swap_chain_support =
                Self::query_swap_chain_support(khr_surface_instance, physical_device, surface)?;
            swap_chain_supported = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        Ok(queue_family_indices.is_complete() && extensions_supported && swap_chain_supported)
    }

    fn query_swap_chain_support(
        khr_surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapChainSupportDetails> {
        let mut swap_chain_support_details = SwapChainSupportDetails::default();

        unsafe {
            swap_chain_support_details.capabilities = khr_surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)?;
            swap_chain_support_details.formats = khr_surface_instance
                .get_physical_device_surface_formats(physical_device, surface)?;
            swap_chain_support_details.present_modes = khr_surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface)?;
        };

        Ok(swap_chain_support_details)
    }

    fn find_queue_families(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndices> {
        let mut queue_family_indices = QueueFamilyIndices::default();
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (index, properties) in queue_family_properties.iter().enumerate() {
            if properties.queue_flags.intersects(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            let present_support = unsafe {
                khr_surface_instance.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )?
            };

            if present_support {
                queue_family_indices.present_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        Ok(queue_family_indices)
    }

    fn create_logical_device(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
        let queue_family_indices =
            Self::find_queue_families(instance, khr_surface_instance, physical_device, surface)?;

        let queue_create_infos =
            if queue_family_indices.graphics_family == queue_family_indices.present_family {
                vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_priorities(std::slice::from_ref(&1.0))
                        .queue_family_index(queue_family_indices.graphics_family.unwrap()),
                ]
            } else {
                vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_priorities(std::slice::from_ref(&1.0))
                        .queue_family_index(queue_family_indices.graphics_family.unwrap()),
                    vk::DeviceQueueCreateInfo::default()
                        .queue_priorities(std::slice::from_ref(&1.0))
                        .queue_family_index(queue_family_indices.present_family.unwrap()),
                ]
            };
        let device_features = vk::PhysicalDeviceFeatures::default();
        let extensions = ["VK_KHR_swapchain".to_string()];
        let extensions: Vec<CString> = extensions
            .iter()
            .map(|ext| CString::new(ext.as_str()).unwrap())
            .collect();
        let extensions: Vec<*const c_char> = extensions
            .iter()
            .map(|ext| ext.as_ptr() as *const c_char)
            .collect();

        let device_create_info = vk::DeviceCreateInfo::default()
            .enabled_features(&device_features)
            .enabled_extension_names(&extensions)
            .queue_create_infos(&queue_create_infos);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .context("Failed to create logical device")?
        };

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_family.unwrap(), 0) };

        Ok((device, graphics_queue, present_queue))
    }

    #[allow(clippy::type_complexity)]
    fn create_swap_chain(
        window: &glfw::PWindow,
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        khr_swapchain_device: &khr::swapchain::Device,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surface: vk::SurfaceKHR,
    ) -> Result<(
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    )> {
        let swap_chain_support =
            Self::query_swap_chain_support(khr_surface_instance, physical_device, surface)?;

        let surface_format = swap_chain_support
            .formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(swap_chain_support.formats.first().unwrap());

        let present_mode = swap_chain_support
            .present_modes
            .into_iter()
            .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let extent = if swap_chain_support.capabilities.current_extent.width != u32::MAX {
            swap_chain_support.capabilities.current_extent
        } else {
            let (width, height) = window.get_framebuffer_size();
            let width = width as u32;
            let height = height as u32;

            vk::Extent2D::default()
                .width(width.clamp(
                    swap_chain_support.capabilities.min_image_extent.width,
                    swap_chain_support.capabilities.max_image_extent.width,
                ))
                .height(height.clamp(
                    swap_chain_support.capabilities.min_image_extent.height,
                    swap_chain_support.capabilities.max_image_extent.height,
                ))
        };

        let mut image_count = swap_chain_support.capabilities.min_image_count + 1;

        if swap_chain_support.capabilities.max_image_count > 0
            && image_count > swap_chain_support.capabilities.max_image_count
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        // NOTE: `image_array_layers` is always `1` unless you develop a stereoscopic 3D
        // application
        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swap_chain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let queue_family_indices =
            Self::find_queue_families(instance, khr_surface_instance, physical_device, surface)?;

        let queue_family_indices_slice = [
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.present_family.unwrap(),
        ];
        swapchain_create_info =
            if queue_family_indices.graphics_family != queue_family_indices.present_family {
                swapchain_create_info
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&queue_family_indices_slice)
            } else {
                swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

        let swapchain =
            unsafe { khr_swapchain_device.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { khr_swapchain_device.get_swapchain_images(swapchain)? };
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());
        for image in swapchain_images.iter() {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let image_view = unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .context("Filed to create image view")?
            };

            swapchain_image_views.push(image_view);
        }

        Ok((
            swapchain,
            swapchain_images,
            swapchain_image_views,
            surface_format.format,
            extent,
        ))
    }

    fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
        let color_attachement = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attackment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(slice::from_ref(&color_attackment_ref));

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(slice::from_ref(&color_attachement))
            .subpasses(slice::from_ref(&subpass));

        let render_pass = unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .context("Failed to create render pass")?
        };

        Ok(render_pass)
    }

    fn create_graphics_pipeline(device: &ash::Device) -> Result<vk::PipelineLayout> {
        // Create vertex and fragment shaders
        {
            let vertex_shader_spirv = include_bytes!("../shaders/shader.spirv.vert");
            let fragment_shader_spirv = include_bytes!("../shaders/shader.spirv.frag");

            let vertex_shader_module = Self::create_shader_module(device, vertex_shader_spirv)?;
            let fragment_shader_module = Self::create_shader_module(device, fragment_shader_spirv)?;

            let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(c"main");

            let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(c"main");

            let _shader_stages = [
                vertex_shader_stage_create_info,
                fragment_shader_stage_create_info,
            ];

            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }
        }

        // Vertex input
        {
            let _vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&[])
                .vertex_attribute_descriptions(&[]);
        }

        // Input assembly
        {
            let _input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
        }

        // Viewport
        {
            let _viewport_create_info = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
        }

        // Rasterizer
        {
            let _rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
        }

        // Multisampling
        {
            let _multisample_create_info = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        }

        // Color blending
        {
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);

            let _color_blending_create_info = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&[color_blend_attachment])
                .blend_constants([0.0, 0.0, 0.0, 0.0]);
        }

        // Dynamic state
        {
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let _dynamic_state_create_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        }

        let pipeline_layout = {
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[])
                .push_constant_ranges(&[]);

            unsafe {
                device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .context("Failed to create pipeline layout")
            }?
        };

        Ok(pipeline_layout)
    }

    fn create_shader_module(device: &ash::Device, spirv_data: &[u8]) -> Result<vk::ShaderModule> {
        let layout = Layout::from_size_align(spirv_data.len(), 4)?;
        let pointer = unsafe { alloc(layout) };

        if pointer.is_null() {
            anyhow::bail!("Unable to allocate 4 byte alligned buffer");
        }
        let slice = unsafe { slice::from_raw_parts_mut(pointer, layout.size()) };
        slice.copy_from_slice(spirv_data);

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
            let pointer: *const u32 = slice.as_ptr().cast();
            let len = slice.len() / 4;
            slice::from_raw_parts(pointer, len)
        });
        let shader_module =
            unsafe { device.create_shader_module(&shader_module_create_info, None) }?;

        Ok(shader_module)
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_utils_instance) = self.debug_utils_instance.as_ref() {
                if let Some(debug_utils_messanger) = self.debug_utils_messanger {
                    debug_utils_instance.destroy_debug_utils_messenger(debug_utils_messanger, None);
                }
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.swapchain_images_views
                .iter()
                .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.khr_swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.khr_surface_instance
                .destroy_surface(self.window_surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        };
    }
}
