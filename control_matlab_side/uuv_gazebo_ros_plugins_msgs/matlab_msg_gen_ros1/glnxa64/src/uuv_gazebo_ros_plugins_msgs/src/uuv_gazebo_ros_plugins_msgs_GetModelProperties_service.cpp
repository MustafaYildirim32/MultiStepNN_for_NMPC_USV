// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/GetModelPropertiesRequest
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
#include "uuv_gazebo_ros_plugins_msgs/GetModelProperties.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetModelPropertiesRequest");
    }
    return std::move(outArray);
  }
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //link_names
        const matlab::data::CellArray link_names_cellarr = arr["LinkNames"];
        size_t nelem = link_names_cellarr.getNumberOfElements();
        for (size_t idx=0; idx < nelem; ++idx){
        	const matlab::data::CharArray link_names_arr = link_names_cellarr[idx];
        	msg->link_names.push_back(link_names_arr.toAscii());
        }
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'LinkNames' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'LinkNames' is wrong type; expected a string.");
    }
    try {
        //models
        const matlab::data::StructArray models_arr = arr["Models"];
        for (auto _modelsarr : models_arr) {
        	uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel _val;
        auto msgClassPtr_models = getCommonObject<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel>("uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common",loader);
        msgClassPtr_models->copy_from_struct(&_val,_modelsarr,loader);
        	msg->models.push_back(_val);
        }
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Models' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Models' is wrong type; expected a struct.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","LinkNames","Models"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetModelPropertiesResponse");
    // link_names
    auto currentElement_link_names = (msg + ctr)->link_names;
    auto link_namesoutCell = factory.createCellArray({currentElement_link_names.size(),1});
    for(size_t idxin = 0; idxin < currentElement_link_names.size(); ++ idxin){
    	link_namesoutCell[idxin] = factory.createCharArray(currentElement_link_names[idxin]);
    }
    outArray[ctr]["LinkNames"] = link_namesoutCell;
    // models
    auto currentElement_models = (msg + ctr)->models;
    auto msgClassPtr_models = getCommonObject<uuv_gazebo_ros_plugins_msgs::UnderwaterObjectModel>("uuv_gazebo_ros_plugins_msgs_msg_UnderwaterObjectModel_common",loader);
    outArray[ctr]["Models"] = msgClassPtr_models->get_arr(factory,&currentElement_models[0],loader,currentElement_models.size());
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_GetModelProperties_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_GetModelProperties_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_GetModelProperties_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          uuv_gazebo_ros_plugins_msgs_GetModelProperties_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request,uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response,uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          uuv_gazebo_ros_plugins_msgs_GetModelProperties_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request,uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          uuv_gazebo_ros_plugins_msgs_GetModelProperties_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<uuv_gazebo_ros_plugins_msgs::GetModelProperties,uuv_gazebo_ros_plugins_msgs::GetModelProperties::Request,uuv_gazebo_ros_plugins_msgs::GetModelProperties::Response,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          uuv_gazebo_ros_plugins_msgs_GetModelProperties_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetModelPropertiesRequest,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse,uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesRequest_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetModelPropertiesRequest>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetModelPropertiesResponse_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetModelPropertiesResponse>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_GetModelProperties_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
