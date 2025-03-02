// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/GetThrusterStateRequest
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
#include "uuv_gazebo_ros_plugins_msgs/GetThrusterState.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetThrusterStateRequest");
    }
    return std::move(outArray);
  }
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //is_on
        const matlab::data::TypedArray<bool> is_on_arr = arr["IsOn"];
        msg->is_on = is_on_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'IsOn' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'IsOn' is wrong type; expected a logical.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","IsOn"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetThrusterStateResponse");
    // is_on
    auto currentElement_is_on = (msg + ctr)->is_on;
    outArray[ctr]["IsOn"] = factory.createScalar(static_cast<bool>(currentElement_is_on));
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_GetThrusterState_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_GetThrusterState_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterState_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterState_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response,uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterState_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterState_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterState,uuv_gazebo_ros_plugins_msgs::GetThrusterState::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterState::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterState_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterStateRequest,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterStateResponse,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateRequest_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterStateRequest>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetThrusterStateResponse_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterStateResponse>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_GetThrusterState_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
