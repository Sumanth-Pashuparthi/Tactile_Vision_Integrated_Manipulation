#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <string>
#include <ctime>

class CombinedPointCloudProcessor : public rclcpp::Node
{
public:
    CombinedPointCloudProcessor() : Node("combined_pointcloud_processor")
    {
        pointcloud_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/realsense/points", 10,
            std::bind(&CombinedPointCloudProcessor::pointCloudCallback, this, std::placeholders::_1));

        transformed_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_pointcloud_data", 10);
        processed_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("segmented_plane", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
    {
        // Convert ROS PointCloud2 message to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // Get transform
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer_->lookupTransform("world", "camera_link", tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not transform: %s", ex.what());
            return;
        }

        // Transform the point cloud
        Eigen::Affine3d transform = tf2::transformToEigen(transform_stamped.transform);

        RCLCPP_INFO(this->get_logger(), "Transform Translation: [%.3f, %.3f, %.3f]",
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z);
        RCLCPP_INFO(this->get_logger(), "Transform Rotation: [%.3f, %.3f, %.3f, %.3f]",
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z,
        transform_stamped.transform.rotation.w);

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*transformed_cloud, centroid);
        RCLCPP_INFO(this->get_logger(), "Transformed cloud centroid: [%.3f, %.3f, %.3f]",
        centroid[0], centroid[1], centroid[2]);

        // Publish transformed cloud
        sensor_msgs::msg::PointCloud2 transformed_cloud_msg;
        pcl::toROSMsg(*transformed_cloud, transformed_cloud_msg);
        transformed_cloud_msg.header.frame_id = "world";
        transformed_cloud_msg.header.stamp = this->get_clock()->now();
        transformed_pointcloud_publisher_->publish(transformed_cloud_msg);

        // Apply VoxelGrid filter to downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(transformed_cloud);
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);  // 1cm grid
        voxel_filter.filter(*cloud_downsampled);

        // Apply PassThrough filters for all axes
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Filter Z axis
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(cloud_downsampled);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(-0.05, std::numeric_limits<float>::max());
        pass_z.filter(*cloud_filtered);

        // Filter X axis
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(cloud_filtered);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(-0.5, 0.75);
        pass_x.filter(*cloud_filtered);

        // // Filter Y axis
        // pcl::PassThrough<pcl::PointXYZ> pass_y;
        // pass_y.setInputCloud(cloud_filtered);
        // pass_y.setFilterFieldName("y");
        // pass_y.setFilterLimits(-0.5, 2.5);
        // pass_y.filter(*cloud_filtered);

        RCLCPP_INFO(this->get_logger(), "Filtered cloud has %ld points after xyz filtering", cloud_filtered->size());

        // Plane segmentation
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not estimate a planar model for the given dataset.");
            return;
        }

        // Remove plane
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_without_plane);

                    // Add after plane segmentation, before removing the plane
        RCLCPP_INFO(this->get_logger(), "Plane coefficients: [%.3f, %.3f, %.3f, %.3f]",
        coefficients->values[0],
        coefficients->values[1],
        coefficients->values[2],
        coefficients->values[3]);

        // Perform clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_without_plane);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.0001);     // 2cm distance between points
        ec.setMinClusterSize(0);        // Minimum number of points
        ec.setMaxClusterSize(150);      // Maximum number of points
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_without_plane);
        ec.extract(cluster_indices);

        // Combine valid clusters
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        for (const auto& cluster : cluster_indices)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster.indices)
            {
                cloud_cluster->push_back((*cloud_without_plane)[idx]);
            }
            
            // RCLCPP_INFO(this->get_logger(), "Cluster size: %lu points", cloud_cluster->size());
            *final_cloud += *cloud_cluster;
        }

        // RCLCPP_INFO(this->get_logger(), "Found %lu clusters", cluster_indices.size());
        // RCLCPP_INFO(this->get_logger(), "Final cloud has %ld points after clustering", final_cloud->size());

        // Publish final result
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*final_cloud, output);
        output.header.frame_id = "world";
        output.header.stamp = this->get_clock()->now();
        Eigen::Vector4f final_centroid;
        pcl::compute3DCentroid(*final_cloud, final_centroid);
        RCLCPP_INFO(this->get_logger(), "Final segmented cloud centroid: [%.3f, %.3f, %.3f]",
        final_centroid[0], final_centroid[1], final_centroid[2]);
        processed_pointcloud_publisher_->publish(output);
    }
    void savePCDFile(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& prefix) {
    // Get current timestamp
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    
    // Create filename with timestamp
    std::string filename = prefix + "_" + timestamp + ".pcd";
    
    if (pcl::io::savePCDFileASCII(filename, *cloud) == -1) {
        RCLCPP_ERROR(this->get_logger(), "Failed to save PCD file: %s", filename.c_str());
    } else {
        RCLCPP_INFO(this->get_logger(), "Saved PCD file: %s", filename.c_str());
    }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr transformed_pointcloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_pointcloud_publisher_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    // Save transformed cloud
    savePCDFile(transformed_cloud, "transformed_cloud");
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CombinedPointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}