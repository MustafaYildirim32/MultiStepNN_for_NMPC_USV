// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/FloatStamped
#include "boost/date_time.hpp"
#include "boost/shared_array.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4265)
#pragma warning(disable : 4458)
#pragma warning(disable : 4100)
#pragma warning(disable : 4127)
#pragma warning(disable : 4267)
#pragma warning(disable : 4068)
#pragma warning(disable : 4245)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif //_MSC_VER
#include "ros/ros.h"
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"
#include "visibility_control.h"
#include "MATLABROSMsgInterface.hpp"
#include "ROSPubSubTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::FloatStamped> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::FloatStamped* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::FloatStamped* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::FloatStamped* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //header
        const matlab::data::StructArray header_arr = arr["Header"];
        auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
        msgClassPtr_header->copy_from_struct(&msg->header,header_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Header' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Header' is wrong type; expected a struct.");
    }
    try {
        //data
        const matlab::data::TypedArray<double> data_arr = arr["Data"];
        msg->data = data_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Data' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Data' is wrong type; expected a double.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::FloatStamped* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Header","Data"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/FloatStamped");
    // header
    auto currentElement_header = (msg + ctr)->header;
    auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
    outArray[ctr]["Header"] = msgClassPtr_header->get_arr(factory, &currentElement_header, loader);
    // data
    auto currentElement_data = (msg + ctr)->data;
    outArray[ctr]["Data"] = factory.createScalar(currentElement_data);
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_FloatStamped_message : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_FloatStamped_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_FloatStamped_message::generatePublisherInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::FloatStamped,uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         uuv_gazebo_ros_plugins_msgs_FloatStamped_message::generateSubscriberInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::FloatStamped,uuv_gazebo_ros_plugins_msgs::FloatStamped::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface>
         uuv_gazebo_ros_plugins_msgs_FloatStamped_message::generateRosbagWriterInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::FloatStamped,uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common>>();
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_FloatStamped_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::FloatStamped>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_FloatStamped_message, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1