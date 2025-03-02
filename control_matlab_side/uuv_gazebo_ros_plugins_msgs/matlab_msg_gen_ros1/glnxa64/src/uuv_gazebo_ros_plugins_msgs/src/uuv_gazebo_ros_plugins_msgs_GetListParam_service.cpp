// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/GetListParamRequest
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
#include "uuv_gazebo_ros_plugins_msgs/GetListParam.h"
#include "visibility_control.h"
#include "ROSPubSubTemplates.hpp"
#include "ROSServiceTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetListParam::Request> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetListParam::Request* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetListParam::Request* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetListParam::Request* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetListParam::Request* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetListParamRequest");
    }
    return std::move(outArray);
  }
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetListParam::Response> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetListParam::Response* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetListParam::Response* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::GetListParam::Response* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //description
        const matlab::data::CharArray description_arr = arr["Description"];
        msg->description = description_arr.toAscii();
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Description' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Description' is wrong type; expected a string.");
    }
    try {
        //tags
        const matlab::data::CellArray tags_cellarr = arr["Tags"];
        size_t nelem = tags_cellarr.getNumberOfElements();
        for (size_t idx=0; idx < nelem; ++idx){
        	const matlab::data::CharArray tags_arr = tags_cellarr[idx];
        	msg->tags.push_back(tags_arr.toAscii());
        }
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Tags' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Tags' is wrong type; expected a string.");
    }
    try {
        //data
        const matlab::data::TypedArray<double> data_arr = arr["Data"];
        size_t nelem = data_arr.getNumberOfElements();
        	msg->data.resize(nelem);
        	std::copy(data_arr.begin(), data_arr.begin()+nelem, msg->data.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Data' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Data' is wrong type; expected a double.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::GetListParam::Response* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Description","Tags","Data"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/GetListParamResponse");
    // description
    auto currentElement_description = (msg + ctr)->description;
    outArray[ctr]["Description"] = factory.createCharArray(currentElement_description);
    // tags
    auto currentElement_tags = (msg + ctr)->tags;
    auto tagsoutCell = factory.createCellArray({currentElement_tags.size(),1});
    for(size_t idxin = 0; idxin < currentElement_tags.size(); ++ idxin){
    	tagsoutCell[idxin] = factory.createCharArray(currentElement_tags[idxin]);
    }
    outArray[ctr]["Tags"] = tagsoutCell;
    // data
    auto currentElement_data = (msg + ctr)->data;
    outArray[ctr]["Data"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::GetListParam::Response::_data_type::const_iterator, double>({currentElement_data.size(),1}, currentElement_data.begin(), currentElement_data.end());
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_GetListParam_service : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_GetListParam_service(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
    virtual std::shared_ptr<MATLABSvcServerInterface> generateSvcServerInterface();
    virtual std::shared_ptr<MATLABSvcClientInterface> generateSvcClientInterface();
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_GetListParam_service::generatePublisherInterface(ElementType type){
    std::shared_ptr<MATLABPublisherInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetListParam::Request,uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::GetListParam::Response,uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSubscriberInterface> 
          uuv_gazebo_ros_plugins_msgs_GetListParam_service::generateSubscriberInterface(ElementType type){
    std::shared_ptr<MATLABSubscriberInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetListParam::Request,uuv_gazebo_ros_plugins_msgs::GetListParam::Request::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::GetListParam::Response,uuv_gazebo_ros_plugins_msgs::GetListParam::Response::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
  std::shared_ptr<MATLABSvcServerInterface> 
          uuv_gazebo_ros_plugins_msgs_GetListParam_service::generateSvcServerInterface(){
    return std::make_shared<ROSSvcServerImpl<uuv_gazebo_ros_plugins_msgs::GetListParam::Request,uuv_gazebo_ros_plugins_msgs::GetListParam::Response,uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common>>();
  }
  std::shared_ptr<MATLABSvcClientInterface> 
          uuv_gazebo_ros_plugins_msgs_GetListParam_service::generateSvcClientInterface(){
    return std::make_shared<ROSSvcClientImpl<uuv_gazebo_ros_plugins_msgs::GetListParam,uuv_gazebo_ros_plugins_msgs::GetListParam::Request,uuv_gazebo_ros_plugins_msgs::GetListParam::Response,uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common,uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface> 
          uuv_gazebo_ros_plugins_msgs_GetListParam_service::generateRosbagWriterInterface(ElementType type){
    std::shared_ptr<MATLABRosbagWriterInterface> ptr;
    if(type == eRequest){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetListParamRequest,uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common>>();
    }else if(type == eResponse){
        ptr = std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::GetListParamResponse,uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common>>();
    }else{
        throw std::invalid_argument("Wrong input, Expected 'Request' or 'Response'");
    }
    return ptr;
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetListParamRequest_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetListParamRequest>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_GetListParamResponse_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::GetListParamResponse>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_GetListParam_service, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1
