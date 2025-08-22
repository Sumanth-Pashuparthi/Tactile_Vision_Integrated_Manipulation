#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class PointCloudMerger : public rclcpp::Node
{
public:
    PointCloudMerger() : Node("pointcloud_merger") 
    {
        // Declare parameters with optimized values for can reconstruction
        this->declare_parameter("separation_distance", 0.0);    
        this->declare_parameter("voxel_size", 0.002);          // 2mm voxel size
        this->declare_parameter("outlier_mean_k", 50);         // Increased for better averaging
        this->declare_parameter("outlier_std_dev", 1.0);       // Less aggressive filtering
        this->declare_parameter("rotation_angle", M_PI);       // Try different angles if needed
        this->declare_parameter("height_offset", 0.1);         // Adjustable height offset
        this->declare_parameter("min_points_threshold", 100);  // Minimum points to process

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        sub1_.subscribe(this, "segmented_plane");
        sub2_.subscribe(this, "segmented_plane2");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(20), sub1_, sub2_);
        sync_->registerCallback(
            std::bind(&PointCloudMerger::cloudCallback, this,
                     std::placeholders::_1, std::placeholders::_2));

        merged_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>
            ("merged_pointcloud", 10);

        publish_static_transforms();
    }

private:
    void publish_static_transforms() {
        auto transform = geometry_msgs::msg::TransformStamped();
        transform.header.stamp = this->get_clock()->now();
        transform.header.frame_id = "base_link";
        transform.child_frame_id = "world";
        transform.transform.translation.x = 0.0;
        transform.transform.translation.y = 0.0;
        transform.transform.translation.z = 0.0;
        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = 0.0;
        transform.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(transform);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocess_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        if (cloud->empty()) {
            return cloud;
        }

        // Get parameters
        float voxel_size = this->get_parameter("voxel_size").as_double();
        int outlier_mean_k = this->get_parameter("outlier_mean_k").as_int();
        float outlier_std_dev = this->get_parameter("outlier_std_dev").as_double();

        // Remove outliers
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.setInputCloud(cloud);
        sor.setMeanK(outlier_mean_k);
        sor.setStddevMulThresh(outlier_std_dev);
        sor.filter(*filtered);

        // Downsample
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        vox.setInputCloud(filtered);
        vox.setLeafSize(voxel_size, voxel_size, voxel_size);
        vox.filter(*downsampled);

        return downsampled;
    }

    Eigen::Vector3f get_centroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        return centroid.head<3>();
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud1_msg,
                      const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud2_msg)
    {
        publish_static_transforms();

        // Convert ROS messages to PCL clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>());
        
        pcl::fromROSMsg(*cloud1_msg, *cloud1);
        pcl::fromROSMsg(*cloud2_msg, *cloud2);

        // Check minimum points threshold
        // Convert min_points to size_t to match cloud size type
        std::size_t min_points_threshold = static_cast<std::size_t>(this->get_parameter("min_points_threshold").as_int());
        if (cloud1->size() < min_points_threshold || cloud2->size() < min_points_threshold) {
            RCLCPP_WARN(this->get_logger(), "Insufficient points in clouds: %zu and %zu points",
                        cloud1->size(), cloud2->size());
            return;
        }
        // Preprocess clouds
        auto processed1 = preprocess_cloud(cloud1);
        auto processed2 = preprocess_cloud(cloud2);

        // Get centroids
        Eigen::Vector3f centroid1 = get_centroid(processed1);
        Eigen::Vector3f centroid2 = get_centroid(processed2);

        // Calculate the transformation to align cloud2 with cloud1
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

        // Align centroids in XY plane
        transform(0,3) = centroid1.x() - centroid2.x();
        transform(1,3) = centroid1.y() - centroid2.y();
        
        // Z alignment using bottom points plus offset
        Eigen::Vector4f min_pt1, max_pt1, min_pt2, max_pt2;
        pcl::getMinMax3D(*processed1, min_pt1, max_pt1);
        pcl::getMinMax3D(*processed2, min_pt2, max_pt2);
        float height_offset = this->get_parameter("height_offset").as_double();
        transform(2,3) = min_pt1[2] - min_pt2[2] + height_offset;

        // Apply rotation around Z axis
        float angle = this->get_parameter("rotation_angle").as_double();
        Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
        rotation(0,0) = std::cos(angle);
        rotation(0,1) = -std::sin(angle);
        rotation(1,0) = std::sin(angle);
        rotation(1,1) = std::cos(angle);

        // Create temporary cloud for intermediate transform
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*processed2, *temp_cloud, transform);
        
        // Rotate around aligned centroid
        Eigen::Vector3f aligned_centroid = get_centroid(temp_cloud);
        Eigen::Matrix4f to_origin = Eigen::Matrix4f::Identity();
        to_origin.block<3,1>(0,3) = -aligned_centroid;
        
        Eigen::Matrix4f from_origin = Eigen::Matrix4f::Identity();
        from_origin.block<3,1>(0,3) = aligned_centroid;
        
        Eigen::Matrix4f final_transform = from_origin * rotation * to_origin;

        // Apply final transformation
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*temp_cloud, *aligned_cloud2, final_transform);

        // Combine clouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr combined(new pcl::PointCloud<pcl::PointXYZ>);
        *combined = *processed1;
        *combined += *aligned_cloud2;

        // Calculate final dimensions
        pcl::getMinMax3D(*combined, min_pt1, max_pt1);
        float height = max_pt1[2] - min_pt1[2];
        float width = max_pt1[0] - min_pt1[0];
        float depth = max_pt1[1] - min_pt1[1];

        // Publish combined cloud
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*combined, output);
        output.header.frame_id = "world";
        output.header.stamp = this->get_clock()->now();
        merged_cloud_publisher_->publish(output);

        // Log detailed information
        RCLCPP_INFO(this->get_logger(), 
                    "Merged cloud stats:\n"
                    "  Input points: Cloud1=%ld, Cloud2=%ld\n"
                    "  Final points: %ld\n"
                    "  Dimensions (m): H=%.3f, W=%.3f, D=%.3f\n"
                    "  Rotation: %.1f degrees\n"
                    "  Translation: [%.3f, %.3f, %.3f]",
                    processed1->size(), processed2->size(),
                    combined->points.size(),
                    height, width, depth,
                    angle * 180.0 / M_PI,
                    transform(0,3), transform(1,3), transform(2,3));
    }

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub1_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub2_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr merged_cloud_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudMerger>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}