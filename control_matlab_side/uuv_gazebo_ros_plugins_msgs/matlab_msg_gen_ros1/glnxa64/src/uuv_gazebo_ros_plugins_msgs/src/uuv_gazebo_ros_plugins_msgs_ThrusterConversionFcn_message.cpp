// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn
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
#include "uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn.h"
#include "visibility_control.h"
#include "MATLABROSMsgInterface.hpp"
#include "ROSPubSubTemplates.hpp"
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common : public MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn> {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common(){}
    virtual void copy_from_struct(uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn* msg, MultiLibLoader loader, size_t size = 1);
};
  void uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common::copy_from_struct(uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //function_name
        const matlab::data::CharArray function_name_arr = arr["FunctionName"];
        msg->function_name = function_name_arr.toAscii();
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'FunctionName' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'FunctionName' is wrong type; expected a string.");
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
    try {
        //lookup_table_input
        const matlab::data::TypedArray<double> lookup_table_input_arr = arr["LookupTableInput"];
        size_t nelem = lookup_table_input_arr.getNumberOfElements();
        	msg->lookup_table_input.resize(nelem);
        	std::copy(lookup_table_input_arr.begin(), lookup_table_input_arr.begin()+nelem, msg->lookup_table_input.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'LookupTableInput' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'LookupTableInput' is wrong type; expected a double.");
    }
    try {
        //lookup_table_output
        const matlab::data::TypedArray<double> lookup_table_output_arr = arr["LookupTableOutput"];
        size_t nelem = lookup_table_output_arr.getNumberOfElements();
        	msg->lookup_table_output.resize(nelem);
        	std::copy(lookup_table_output_arr.begin(), lookup_table_output_arr.begin()+nelem, msg->lookup_table_output.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'LookupTableOutput' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'LookupTableOutput' is wrong type; expected a double.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common::get_arr(MDFactory_T& factory, const uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","FunctionName","Tags","Data","LookupTableInput","LookupTableOutput"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("uuv_gazebo_ros_plugins_msgs/ThrusterConversionFcn");
    // function_name
    auto currentElement_function_name = (msg + ctr)->function_name;
    outArray[ctr]["FunctionName"] = factory.createCharArray(currentElement_function_name);
    // tags
    auto currentElement_tags = (msg + ctr)->tags;
    auto tagsoutCell = factory.createCellArray({currentElement_tags.size(),1});
    for(size_t idxin = 0; idxin < currentElement_tags.size(); ++ idxin){
    	tagsoutCell[idxin] = factory.createCharArray(currentElement_tags[idxin]);
    }
    outArray[ctr]["Tags"] = tagsoutCell;
    // data
    auto currentElement_data = (msg + ctr)->data;
    outArray[ctr]["Data"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn::_data_type::const_iterator, double>({currentElement_data.size(),1}, currentElement_data.begin(), currentElement_data.end());
    // lookup_table_input
    auto currentElement_lookup_table_input = (msg + ctr)->lookup_table_input;
    outArray[ctr]["LookupTableInput"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn::_lookup_table_input_type::const_iterator, double>({currentElement_lookup_table_input.size(),1}, currentElement_lookup_table_input.begin(), currentElement_lookup_table_input.end());
    // lookup_table_output
    auto currentElement_lookup_table_output = (msg + ctr)->lookup_table_output;
    outArray[ctr]["LookupTableOutput"] = factory.createArray<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn::_lookup_table_output_type::const_iterator, double>({currentElement_lookup_table_output.size(),1}, currentElement_lookup_table_output.begin(), currentElement_lookup_table_output.end());
    }
    return std::move(outArray);
  } 
class UUV_GAZEBO_ROS_PLUGINS_MSGS_EXPORT uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message::generatePublisherInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSPublisherImpl<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn,uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message::generateSubscriberInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSSubscriberImpl<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn,uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn::ConstPtr,uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface>
         uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message::generateRosbagWriterInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSBagWriterImpl<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn,uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common>>();
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_msg_ThrusterConversionFcn_common, MATLABROSMsgInterface<uuv_gazebo_ros_plugins_msgs::ThrusterConversionFcn>)
CLASS_LOADER_REGISTER_CLASS(uuv_gazebo_ros_plugins_msgs_ThrusterConversionFcn_message, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1