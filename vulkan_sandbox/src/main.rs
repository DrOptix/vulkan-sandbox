use std::{
    alloc::{Layout, alloc},
    borrow::Cow,
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    path::Path,
    ptr, slice,
};

use anyhow::{Context, Result, anyhow};
use ash::{ext::debug_utils, khr, vk};
use glfw::{Glfw, GlfwReceiver};
use image::EncodableLayout;

use engine::log;

fn main() {
    env_logger::init();

    match HelloTriangleApplication::new(800, 600, "Hello triangle") {
        Ok(mut app) => {
            if let Err(err) = app.run() {
                log::error_details(&err);
            }
        }
        Err(err) => {
            log::error_details(&err);
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vertex {
    pos: glam::Vec3,
    color: glam::Vec3,
    tex_coord: glam::Vec2,
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl std::hash::Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        let mut attribute_descriptions = vec![vk::VertexInputAttributeDescription::default(); 3];

        attribute_descriptions[0] = attribute_descriptions[0]
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(Vertex, pos) as u32);

        attribute_descriptions[1] = attribute_descriptions[1]
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(Vertex, color) as u32);

        attribute_descriptions[2] = attribute_descriptions[2]
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(std::mem::offset_of!(Vertex, tex_coord) as u32);

        attribute_descriptions
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
struct UniformBufferObject {
    model: glam::Mat4,
    view: glam::Mat4,
    projection: glam::Mat4,
}

struct HelloTriangleApplication {
    start_time: std::time::Instant,
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
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    framebuffer_resized: bool,
    indices: Vec<u32>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut std::ffi::c_void>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    msaa_samples: vk::SampleCountFlags,
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
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

        Self::log_physical_devices(&instance)?;

        let (physical_device, msaa_samples) =
            Self::pick_physical_device(&instance, &khr_surface_instance, window_surface)?;

        log::debug!("MSAA samples: {msaa_samples:?}");

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

        let render_pass = Self::create_render_pass(
            &instance,
            physical_device,
            &device,
            swapchain_format,
            msaa_samples,
        )?;

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;

        let (pipeline_layout, pipeline) = Self::create_graphics_pipeline(
            &device,
            render_pass,
            descriptor_set_layout,
            msaa_samples,
        )?;

        let command_pool = Self::create_command_pool(
            &instance,
            &khr_surface_instance,
            physical_device,
            &device,
            window_surface,
        )?;

        let (color_image, color_image_memory, color_image_view) = Self::create_color_resource(
            &instance,
            physical_device,
            &device,
            swapchain_extent.width,
            swapchain_extent.height,
            msaa_samples,
            swapchain_format,
        )?;

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resource(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            swapchain_extent,
            msaa_samples,
        )?;

        let swapchain_framebuffers = Self::create_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            color_image_view,
            depth_image_view,
            swapchain_extent,
        )?;

        let (texture_image, texture_image_memory, mip_levels) = Self::create_texture_image(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            Path::new("./textures/viking_room.png"),
        )?;

        let texture_image_view =
            Self::create_texture_image_view(&device, texture_image, mip_levels)?;

        let texture_sampler =
            Self::create_texture_sampler(&instance, &device, physical_device, mip_levels)?;

        let (vertices, indices) = Self::load_model(Path::new("./models/viking_room.obj"))?;

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            &vertices,
        )?;

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            &indices,
        )?;

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            Self::create_uniform_buffers(&instance, physical_device, &device)?;

        let descriptor_pool = Self::create_descriptor_pool(&device)?;

        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            descriptor_set_layout,
            descriptor_pool,
            &uniform_buffers,
            texture_image_view,
            texture_sampler,
        )?;

        let command_buffers = Self::create_command_buffers(&device, command_pool)?;

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device, swapchain_images.len())?;

        Ok(Self {
            start_time: std::time::Instant::now(),
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
            physical_device,
            device,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_framebuffers,
            swapchain_format,
            swapchain_extent,
            render_pass,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            framebuffer_resized: false,
            indices,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
            descriptor_pool,
            descriptor_sets,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            depth_image,
            depth_image_view,
            depth_image_memory,
            msaa_samples,
            color_image,
            color_image_memory,
            color_image_view,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        self.window.set_key_polling(true);
        self.window.set_framebuffer_size_polling(true);

        while !self.window.should_close() {
            self.glfw.poll_events();

            for (_, event) in glfw::flush_messages(&self.window_events) {
                log::debug!("{event:?}");

                if let glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) = event
                {
                    self.window.set_should_close(true)
                } else if let glfw::WindowEvent::FramebufferSize(_, _) = event {
                    self.framebuffer_resized = true;
                }
            }

            self.draw_frame()?;
        }

        unsafe { self.device.device_wait_idle()? };

        Ok(())
    }
}

/// Internal functions
impl HelloTriangleApplication {
    const MAX_FRAMES_IN_FLIGHT: u32 = 2;

    fn create_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        buffer_size: vk::DeviceSize,
        buffer_usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&create_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = Self::find_memory_type(
            instance,
            physical_device,
            memory_requirements.memory_type_bits,
            memory_properties,
        )?;
        let buffer_memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(memory_requirements.size)
                        .memory_type_index(memory_type_index),
                    None,
                )
                .context("Failed to allocate vertex buffer memory")?
        };

        unsafe {
            device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        }

        Ok((buffer, buffer_memory))
    }

    fn create_texture_image(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        path: &Path,
    ) -> Result<(vk::Image, vk::DeviceMemory, u32)> {
        let image = image::open(path).context(format!("Failed to load texture: {path:?}"))?;
        let image = image.to_rgba8();
        let width = image.width();
        let height = image.height();
        let image_size = (4 * width * height) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mip_levels = (std::cmp::max(width, height) as f32).log2().floor() as u32;

        unsafe {
            let data_ptr = device.map_memory(
                staging_buffer_memory,
                0,
                image_size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                image.as_raw().as_bytes().as_ptr(),
                data_ptr as _,
                image.as_bytes().len(),
            );
            device.unmap_memory(staging_buffer_memory);
        };

        let (image, image_memory) = Self::create_image(
            instance,
            physical_device,
            device,
            width,
            height,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::transition_image_layout(
            device,
            command_pool,
            queue,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        )?;

        Self::copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging_buffer,
            image,
            width,
            height,
        )?;

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        Self::generate_mipmaps(
            instance,
            physical_device,
            device,
            command_pool,
            queue,
            image,
            vk::Format::R8G8B8A8_SRGB,
            width,
            height,
            mip_levels,
        )?;

        Ok((image, image_memory, mip_levels))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mipmaps(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<()> {
        let format_properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };

        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            anyhow::bail!("Texture image format does not support linear blitting");
        }

        let command_buffer = Self::begin_single_time_commands(device, command_pool)?;

        let mut memory_barier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .level_count(1),
            );

        let mut mip_width = width;
        let mut mip_height = height;

        for i in 1..mip_levels {
            memory_barier.subresource_range.base_mip_level = i - 1;
            memory_barier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            memory_barier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            memory_barier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            memory_barier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[memory_barier],
                );
            }

            let new_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let new_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(mip_width as i32)
                        .y(mip_height as i32)
                        .z(1),
                ])
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D::default().x(0).y(0).z(0),
                    vk::Offset3D::default()
                        .x(new_mip_width as i32)
                        .y(new_mip_height as i32)
                        .z(1),
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            unsafe {
                device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                )
            };

            memory_barier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            memory_barier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            memory_barier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            memory_barier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[memory_barier],
                );
            }

            mip_width = new_mip_width;
            mip_height = new_mip_height;
        }

        memory_barier.subresource_range.base_mip_level = mip_levels - 1;
        memory_barier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        memory_barier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        memory_barier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        memory_barier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[memory_barier],
            );
        }

        Self::end_single_time_commands(device, command_pool, queue, command_buffer)
    }

    fn create_color_resource(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        msaa_format: vk::Format,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let (image, image_memory) = Self::create_image(
            instance,
            physical_device,
            device,
            width,
            height,
            1,
            msaa_samples,
            msaa_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let image_view =
            Self::create_image_view(device, image, msaa_format, vk::ImageAspectFlags::COLOR, 1)?;

        Ok((image, image_memory, image_view))
    }

    fn create_depth_resource(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        swapchain_extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let depth_format = Self::find_depth_format(instance, physical_device)?;

        let (depth_image, depth_image_memory) = Self::create_image(
            instance,
            physical_device,
            device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let depth_image_view = Self::create_image_view(
            device,
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        Self::transition_image_layout(
            device,
            command_pool,
            queue,
            depth_image,
            depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1,
        )?;

        Ok((depth_image, depth_image_memory, depth_image_view))
    }

    fn find_depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::Format> {
        let supported_format = Self::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;

        Ok(supported_format)
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn find_supported_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format> {
        for candidate in candidates {
            let candidate_props = unsafe {
                instance.get_physical_device_format_properties(physical_device, *candidate)
            };

            let linear_supported = tiling == vk::ImageTiling::LINEAR
                && candidate_props.linear_tiling_features.intersects(features);
            let optimal_supported = tiling == vk::ImageTiling::OPTIMAL
                && candidate_props.optimal_tiling_features.intersects(features);

            if linear_supported || optimal_supported {
                return Ok(*candidate);
            }
        }

        anyhow::bail!("Failed to find supported format");
    }

    fn create_texture_image_view(
        device: &ash::Device,
        image: vk::Image,
        mip_levels: u32,
    ) -> Result<vk::ImageView> {
        let texture_image_view = Self::create_image_view(
            device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;
        Ok(texture_image_view)
    }

    fn create_texture_sampler(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        mip_levels: u32,
    ) -> Result<vk::Sampler> {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);

        let sampler = unsafe {
            device
                .create_sampler(&create_info, None)
                .context("Failed to crate texture sampler")?
        };

        Ok(sampler)
    }

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Result<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe {
            device
                .create_image_view(&create_info, None)
                .context("Failed to crate texture image view")?
        };

        Ok(image_view)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_image(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        width: u32,
        height: u32,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D::default().width(width).height(height).depth(1))
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(samples)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&create_info, None)
                .context("Failed to create image")?
        };

        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(Self::find_memory_type(
                instance,
                physical_device,
                memory_requirements.memory_type_bits,
                properties,
            )?);

        let image_memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .context("Failed to allocate image memory")?
        };

        unsafe {
            device.bind_image_memory(image, image_memory, 0)?;
        }

        Ok((image, image_memory))
    }

    fn load_model(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>)> {
        let mut vertex_data = Vec::new();
        let mut index_data = Vec::new();
        let mut unique_vertices = HashMap::new();

        let (models, _materials) =
            tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS).context("Failed to OBJ load file")?;

        models.into_iter().for_each(|model| {
            model.mesh.indices.into_iter().for_each(|idx| {
                let pos_offset = (3 * idx) as usize;
                let tex_coord_offset = (2 * idx) as usize;

                let vertex = Vertex {
                    pos: glam::vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    color: glam::vec3(1.0, 1.0, 1.0),
                    tex_coord: glam::vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    index_data.push(*index as u32);
                } else {
                    let index = vertex_data.len();
                    unique_vertices.insert(vertex, index);
                    vertex_data.push(vertex);
                    index_data.push(index as u32);
                }
            });
        });

        Ok((vertex_data, index_data))
    }

    fn create_vertex_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        vertices: &[Vertex],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(vertices) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), data_ptr as _, vertices.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::copy_buffer(
            device,
            command_pool,
            queue,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        )?;

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        Ok((vertex_buffer, vertex_buffer_memory))
    }

    fn create_index_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        indices: &[u32],
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = std::mem::size_of_val(indices) as vk::DeviceSize;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data_ptr = device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(indices.as_ptr(), data_ptr as _, indices.len());
            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::copy_buffer(
            device,
            command_pool,
            queue,
            staging_buffer,
            index_buffer,
            buffer_size,
        )?;

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        Ok((index_buffer, index_buffer_memory))
    }

    fn create_uniform_buffers(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<(
        Vec<vk::Buffer>,
        Vec<vk::DeviceMemory>,
        Vec<*mut std::ffi::c_void>,
    )> {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;

        let mut uniform_buffers = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT as usize);
        let mut uniform_buffers_memory = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT as usize);
        let mut uniform_buffers_mapped = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT as usize);

        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                instance,
                physical_device,
                device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let uniform_buffer_mapped = unsafe {
                device.map_memory(
                    uniform_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )?
            };

            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
            uniform_buffers_mapped.push(uniform_buffer_mapped);
        }

        Ok((
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
        ))
    }

    fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT)
                .ty(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSize::default()
                .descriptor_count(Self::MAX_FRAMES_IN_FLIGHT)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(Self::MAX_FRAMES_IN_FLIGHT);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .context("Failed to create descriptor pool")?
        };

        Ok(descriptor_pool)
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffers: &[vk::Buffer],
        texture_image_view: vk::ImageView,
        texture_sampler: vk::Sampler,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = [descriptor_set_layout, descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor sets")?
        };

        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i as usize])
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize);
            let buffer_infos = [buffer_info];

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_image_view)
                .sampler(texture_sampler);
            let image_infos = [image_info];

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i as usize])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&buffer_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i as usize])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&image_infos),
            ];

            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        Ok(descriptor_sets)
    }

    fn begin_single_time_commands(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?;
        }

        Ok(command_buffer)
    }

    fn end_single_time_commands(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        unsafe { device.end_command_buffer(command_buffer)? };

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(command_pool, &command_buffers);
        }

        Ok(())
    }

    fn copy_buffer_to_image(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_commands(device, command_pool)?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D::default().width(width).height(height).depth(1));

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        };

        Self::end_single_time_commands(device, command_pool, queue, command_buffer)?;

        Ok(())
    }

    fn copy_buffer(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_commands(device, command_pool)?;

        unsafe {
            device.cmd_copy_buffer(
                command_buffer,
                source,
                destination,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(size)],
            );
        }

        Self::end_single_time_commands(device, command_pool, queue, command_buffer)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn transition_image_layout(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) -> Result<()> {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            if Self::has_stencil_component(format) {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let command_buffer = Self::begin_single_time_commands(device, command_pool)?;
        let mut memory_barier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty());

        let (source_stage, destination_stage) = if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            memory_barier.src_access_mask = vk::AccessFlags::empty();
            memory_barier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            memory_barier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            memory_barier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        } else if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            memory_barier.src_access_mask = vk::AccessFlags::empty();
            memory_barier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;

            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
        } else {
            anyhow::bail!("Unsupported layout transition");
        };

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[memory_barier],
            );
        }

        Self::end_single_time_commands(device, command_pool, queue, command_buffer)?;

        Ok(())
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        filter_type: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        for i in 0..memory_properties.memory_type_count {
            if (filter_type & (1 << i)) != 0
                && (memory_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                return Ok(i);
            }
        }

        anyhow::bail!("Failed to find suitable memory type")
    }

    fn draw_frame(&mut self) -> Result<()> {
        let fences = [self.in_flight_fences[self.current_frame]];
        unsafe {
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .context("Could not wait for fences")?;
        }

        let acquire_result = unsafe {
            self.khr_swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };

        let image_index = match acquire_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(());
            }
            Err(err) => return Err(anyhow!(err)).context("Failed to acquire swapchain image"),
        };

        self.update_uniform_buffer();

        unsafe {
            self.device
                .reset_fences(&fences)
                .context("Could not reset fences")?;
            self.device
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .context("Unable to reset command buffer")?
        };

        self.record_command_buffer(self.command_buffers[self.current_frame], image_index)?;

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit_info],
                    self.in_flight_fences[self.current_frame],
                )
                .context("Failed to submit draw command buffer")?
        };

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.khr_swapchain_device
                .queue_present(self.present_queue, &present_info)
        };

        match present_result {
            Ok(false) => {
                // Do nothing
            }
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                if self.framebuffer_resized {
                    self.framebuffer_resized = false;
                    self.recreate_swapchain()?;
                }
            }
            Err(err) => return Err(anyhow!(err)).context("Failed to present swapchain image"),
        };

        self.current_frame = (self.current_frame + 1) % Self::MAX_FRAMES_IN_FLIGHT as usize;

        Ok(())
    }

    fn update_uniform_buffer(&mut self) {
        let current_time = std::time::Instant::now();

        let duration = current_time - self.start_time;
        let duration = duration.as_secs_f32();

        let model = glam::Mat4::from_rotation_z(duration * f32::to_radians(45.0));
        let view = glam::Mat4::look_at_rh(
            glam::vec3(2.0, 2.0, 2.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 0.0, 1.0),
        );

        let mut projection = glam::Mat4::perspective_rh(
            f32::to_radians(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );

        projection.col_mut(1)[1] *= -1.0;

        let ubo = UniformBufferObject {
            model,
            view,
            projection,
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                &ubo,
                self.uniform_buffers_mapped[self.current_frame] as _,
                1,
            );
        }
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        let (mut width, mut height) = self.window.get_framebuffer_size();
        while width == 0 || height == 0 {
            let (w, h) = self.window.get_framebuffer_size();
            width = w;
            height = h;
            self.glfw.wait_events();
        }

        unsafe {
            self.device.device_wait_idle()?;
        }

        self.cleanup_swapchain();

        let (
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        ) = Self::create_swap_chain(
            &self.window,
            &self.instance,
            &self.khr_surface_instance,
            &self.khr_swapchain_device,
            self.physical_device,
            &self.device,
            self.window_surface,
        )?;

        let (msaa_image, msaa_image_memory, msaa_image_view) = Self::create_color_resource(
            &self.instance,
            self.physical_device,
            &self.device,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            self.msaa_samples,
            swapchain_format,
        )?;

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_resource(
            &self.instance,
            self.physical_device,
            &self.device,
            self.command_pool,
            self.graphics_queue,
            swapchain_extent,
            self.msaa_samples,
        )?;

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;

        self.color_image = msaa_image;
        self.color_image_memory = msaa_image_memory;
        self.color_image_view = msaa_image_view;

        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;

        self.swapchain_framebuffers = Self::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_image_views,
            self.color_image_view,
            self.depth_image_view,
            self.swapchain_extent,
        )?;

        Ok(())
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device.destroy_image_view(self.color_image_view, None);
            self.device.destroy_image(self.color_image, None);
            self.device.free_memory(self.color_image_memory, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.swapchain_framebuffers
                .iter()
                .for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
            self.swapchain_image_views
                .iter()
                .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.khr_swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn create_sync_objects(
        device: &ash::Device,
        swapchain_images_count: usize,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
        let mut image_available_semaphores =
            Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT as usize);
        let mut render_finished_semaphores = Vec::with_capacity(swapchain_images_count);
        let mut in_flight_fences = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT as usize);

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .context("Failed to create image available semaphore")?
            };
            image_available_semaphores.push(semaphore);

            let fence = unsafe {
                device
                    .create_fence(&fence_create_info, None)
                    .context("Failed to create in flight fence")?
            };
            in_flight_fences.push(fence);
        }

        for _ in 0..swapchain_images_count {
            let semaphore = unsafe {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .context("Failed to create render finished semaphore")?
            };
            render_finished_semaphores.push(semaphore);
        }

        Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ))
    }

    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
    ) -> Result<()> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .context("Failed to being recording command buffer")?
        };

        let mut clear_colors = [vk::ClearValue::default(); 2];
        clear_colors[0].color.float32 = [0.0, 0.0, 0.0, 1.0];
        clear_colors[1].depth_stencil = vk::ClearDepthStencilValue::default().depth(1.0).stencil(0);

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index as usize])
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(self.swapchain_extent),
            )
            .clear_values(&clear_colors);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        };

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default().x(0).y(0))
            .extent(self.swapchain_extent);

        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0];
        let descriptor_sets = [self.descriptor_sets[self.current_frame]];

        unsafe {
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &descriptor_sets,
                &[],
            );
            self.device
                .cmd_draw_indexed(command_buffer, self.indices.len() as u32, 1, 0, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .context("Failed to render command buffer")?;
        };

        Ok(())
    }

    fn create_window(
        glfw: &mut Glfw,
        width: u32,
        height: u32,
        title: &str,
    ) -> Result<(glfw::PWindow, GlfwReceiver<(f64, glfw::WindowEvent)>)> {
        log::info!("Creating window '{title}' ({width}, {height})");

        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(true));

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

    fn get_max_usable_sample_count(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::SampleCountFlags {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let counts = properties.limits.framebuffer_color_sample_counts
            & properties.limits.framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, vk::SampleCountFlags)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            anyhow::bail!("Failed to find devices with Vulkan support");
        }

        for physical_device in physical_devices.into_iter() {
            if Self::is_device_suitable(instance, khr_surface_instance, physical_device, surface)? {
                let msaa_samples = Self::get_max_usable_sample_count(instance, physical_device);
                return Ok((physical_device, msaa_samples));
            }
        }

        anyhow::bail!("Unable to find a suitable device");
    }

    fn log_physical_devices(instance: &ash::Instance) -> Result<()> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        physical_devices
            .into_iter()
            .enumerate()
            .for_each(|(i, physical_device)| {
                let physical_device_properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };

                let device_name = unsafe {
                    std::ffi::CStr::from_ptr(physical_device_properties.device_name.as_ptr())
                };

                log::info!("Physical Device {i}: {}", device_name.to_string_lossy());

                let queue_family_properties = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };

                queue_family_properties.into_iter().enumerate().for_each(
                    |(j, queue_family_property)| {
                        log::info!("  Queue Family {j}");
                        log::info!("    Queue Flags: {:?}", queue_family_property.queue_flags);
                        log::info!("    Queue Count: {}", queue_family_property.queue_count);
                        log::info!(
                            "    Timestamp Valid Bits: {}",
                            queue_family_property.timestamp_valid_bits
                        );
                        log::info!(
                            "    Min Image Transfer Granularity: {:?}",
                            queue_family_property.min_image_transfer_granularity
                        );
                    },
                );
            });

        Ok(())
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
        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };

        let mut swap_chain_supported = false;
        if extensions_supported {
            let swap_chain_support =
                Self::query_swap_chain_support(khr_surface_instance, physical_device, surface)?;
            swap_chain_supported = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        Ok(queue_family_indices.is_complete()
            && extensions_supported
            && swap_chain_supported
            && supported_features.sampler_anisotropy > 0)
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
        let device_features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .sample_rate_shading(true);
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
            let image_view = Self::create_image_view(
                device,
                *image,
                surface_format.format,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;
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

    fn create_render_pass(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<vk::RenderPass> {
        let color_attachement = vk::AttachmentDescription::default()
            .format(format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachement = vk::AttachmentDescription::default()
            .format(Self::find_depth_format(instance, physical_device)?)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachement_resolve = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attackment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attackment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attackment_resolve_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(slice::from_ref(&color_attackment_ref))
            .depth_stencil_attachment(&depth_attackment_ref)
            .resolve_attachments(slice::from_ref(&color_attackment_resolve_ref));

        let subpass_dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = [
            color_attachement,
            depth_attachement,
            color_attachement_resolve,
        ];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(slice::from_ref(&subpass))
            .dependencies(slice::from_ref(&subpass_dependency));

        let render_pass = unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .context("Failed to create render pass")?
        };

        Ok(render_pass)
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let ubo_layout = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layouts = [ubo_layout, sampler_layout];
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layouts);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&create_info, None)? };

        Ok(descriptor_set_layout)
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
        // Create vertex and fragment shaders
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

        let shader_stages = [
            vertex_shader_stage_create_info,
            fragment_shader_stage_create_info,
        ];

        // Vertex input
        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_description();
        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        // Input assembly
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport
        let viewport_create_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterizer
        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        // Multisampling
        let multisample_create_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(msaa_samples)
            .sample_shading_enable(true)
            .min_sample_shading(0.2);

        // Color blending
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending_create_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_blend_attachment))
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let descriptor_layouts = [descriptor_set_layout];
        let pipeline_layout = {
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&descriptor_layouts)
                .push_constant_ranges(&[]);

            unsafe {
                device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
                    .context("Failed to create pipeline layout")
            }?
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisample_create_info)
            .color_blend_state(&color_blending_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .depth_stencil_state(&depth_stencil_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null());

        // Not the best thing to to, but I don't the mental energy to deal with it right now, so
        // yeah, let that `unwrap()` in there
        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_create_info),
                    None,
                )
                .unwrap()[0]
        };

        // Cleanup
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        Ok((pipeline_layout, graphics_pipeline))
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

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        swapchain_image_views: &[vk::ImageView],
        msaa_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
        swapchain_extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>> {
        let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());

        for image_view in swapchain_image_views {
            let attachments = [msaa_image_view, depth_image_view, *image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None)? };
            framebuffers.push(framebuffer);
        }

        Ok(framebuffers)
    }

    fn create_command_pool(
        instance: &ash::Instance,
        khr_surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::CommandPool> {
        let queue_family_indices =
            Self::find_queue_families(instance, khr_surface_instance, physical_device, surface)?;
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        Ok(command_pool)
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .context("Failed to allocate command buffers")?
        };

        Ok(command_buffers)
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
            self.cleanup_swapchain();
            self.device.destroy_sampler(self.texture_sampler, None);
            self.device
                .destroy_image_view(self.texture_image_view, None);
            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);
            self.uniform_buffers
                .iter()
                .for_each(|buffer| self.device.destroy_buffer(*buffer, None));
            self.uniform_buffers_memory
                .iter()
                .for_each(|buffer_memory| self.device.free_memory(*buffer_memory, None));
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.render_finished_semaphores
                .iter()
                .for_each(|semaphore| self.device.destroy_semaphore(*semaphore, None));
            self.image_available_semaphores
                .iter()
                .for_each(|semaphore| self.device.destroy_semaphore(*semaphore, None));
            self.in_flight_fences
                .iter()
                .for_each(|fence| self.device.destroy_fence(*fence, None));
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.khr_surface_instance
                .destroy_surface(self.window_surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        };
    }
}
