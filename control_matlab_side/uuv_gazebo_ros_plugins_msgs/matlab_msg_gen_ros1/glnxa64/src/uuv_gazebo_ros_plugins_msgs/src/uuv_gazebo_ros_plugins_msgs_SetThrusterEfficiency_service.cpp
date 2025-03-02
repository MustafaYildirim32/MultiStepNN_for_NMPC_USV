// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiencyRequest
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
#include "uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiency.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //efficiency
        const matlab::data::TypedArray<double> efficiency_arr = arr["Efficiency"];
        msg->efficiency = efficiency_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Efficiency' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Efficiency' is wrong type; expected a double.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Efficiency"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiencyRequest");
    // efficiency
    auto currentElement_efficiency = (msg + ctr)->efficiency;
    outArray[ctr]["Efficiency"] = factory.createScalar(currentElement_efficiency);
    }
    return std::move(outArray);
  }
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //success
        const matlab::data::TypedArray<bool> success_arr = arr["Success"];
        msg->success = success_arr[0];
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Success' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Success' is wrong type; expected a logical.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Success"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/SetThrusterEfficiencyResponse");
    // success
    auto currentElement_success = (msg + ctr)->success;
    outArray[ctr]["Success"] = factory.createScalar(static_cast<bool>(currentElement_success));
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request,uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response,uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request,uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency,uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Request,uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiency::Response,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse,uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyRequest_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyRequest>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_SetThrusterEfficiencyResponse_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::SetThrusterEfficiencyResponse>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_SetThrusterEfficiency_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
