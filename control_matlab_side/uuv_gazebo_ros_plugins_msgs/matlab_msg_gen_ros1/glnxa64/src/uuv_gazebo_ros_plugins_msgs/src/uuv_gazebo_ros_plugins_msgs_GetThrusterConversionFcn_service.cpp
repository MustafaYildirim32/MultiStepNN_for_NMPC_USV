// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest
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
#include "uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcn.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnRequest");
    }
    return std::move(outArray);
  }
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //fcn
        const matlab::data::StructArray fcn_arr = arr["Fcn"];
        auto msgClassPtr_fcn = getCommonObject<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn>("uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common",loader);
        msgClassPtr_fcn->copy_from_struct(&msg->fcn,fcn_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Fcn' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Fcn' is wrong type; expected a struct.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Fcn"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetThrusterConversionFcnResponse");
    // fcn
    auto currentElement_fcn = (msg + ctr)->fcn;
    auto msgClassPtr_fcn = getCommonObject<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn>("uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common",loader);
    outArray[ctr]["Fcn"] = msgClassPtr_fcn->get_arr(factory, &currentElement_fcn, loader);
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response,uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn,uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Request,uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcn::Response,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse,uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnRequest_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnRequest>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetThrusterConversionFcnResponse_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetThrusterConversionFcnResponse>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_GetThrusterConversionFcn_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
