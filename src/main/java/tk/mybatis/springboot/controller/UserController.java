package tk.mybatis.springboot.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import tk.mybatis.springboot.model.User;
import tk.mybatis.springboot.service.UserService;

import java.util.List;

import tk.mybatis.springboot.util.Bayes;

import javax.servlet.http.HttpServletRequest;

@CrossOrigin
@RestController
@RequestMapping("/api")
public class UserController {

    @Autowired
    private UserService UserService;

    private JSONObject attList;

    private Bayes bn;

    private void initAttrList(){
        this.attList = new JSONObject();
        this.attList.put("wei" ,  JSON.parseObject("{\"description\" : \"Data adjustment factor\", \"type\": \"numerical\"}"));
        this.attList.put("gen" ,  JSON.parseObject("{\"description\" : \"gender\", \"type\": \"categorical\"}"));
        this.attList.put("cat" ,  JSON.parseObject("{\"description\" : \"Whether he or she is a Catholic believer?\", \"type\": \"categorical\"}"));
        this.attList.put("res" ,  JSON.parseObject("{\"description\" : \"Residence\", \"type\": \"categorical\"}"));
        this.attList.put("sch" ,  JSON.parseObject("{\"description\" : \"School\", \"type\": \"categorical\"}"));
        this.attList.put("fue" ,  JSON.parseObject("{\"description\" : \"Whether the father is unemployed?\", \"type\": \"categorical\"}"));
        this.attList.put("gcs" ,  JSON.parseObject("{\"description\" : \"Whether he or she has five or more GCSEs at grades AC?\", \"type\": \"categorical\"}"));
        this.attList.put("fmp" ,  JSON.parseObject("{\"description\" : \"Whether the father is at least the management?\", \"type\": \"categorical\"}"));
        this.attList.put("lvb" ,  JSON.parseObject("{\"description\" : \"Whether live with parents?\", \"type\": \"categorical\"}"));
        this.attList.put("tra" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep training within six years?\", \"type\": \"numerical\"}"));
        this.attList.put("emp" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep employment within six years?\", \"type\": \"numerical\"}"));
        this.attList.put("jol" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep joblessness within six years?\", \"type\": \"numerical\"}"));
        this.attList.put("fe" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a further education within six years?\", \"type\": \"numerical\"}"));
        this.attList.put("he" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a higher education within six years?\", \"type\": \"numerical\"}"));
        this.attList.put("ascc" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep at school within six years?\", \"type\": \"numerical\"}"));
    }

    UserController() {
        initAttrList();
        bn = new Bayes();
    }

    @RequestMapping//Home
    public List<User> getAll() {
        return UserService.getAll();
    }

    @RequestMapping(value = "/load_data", method = RequestMethod.POST)
    public String loadData(HttpServletRequest request){
        JSONObject response = new JSONObject();
        response.put("attList",this.attList);
        return response.toJSONString();
    }

    @RequestMapping(value = "set_selected_attribute", method = RequestMethod.POST)
    public void set_selected_attribute(HttpServletRequest request){
        List<JSONObject> selectAtt = JSON.parseArray(request.getParameter("attributes"), JSONObject.class);
        bn.setSelectedAttribute(selectAtt);
    }

    @RequestMapping(value = "/get_attribute_distribution", method = RequestMethod.POST)
    public String get_attribute_distribution(HttpServletRequest request) {
        List<String> selectAtt = JSON.parseArray(request.getParameter("attributes"), String.class);
        JSONArray attributes = new JSONArray();
        for(String att: selectAtt){
            JSONObject attObj = new JSONObject();
            String type = (String)((JSONObject) this.attList.get(att)).get("type");
            JSONArray dataList = bn.getAttDistribution(att, type);
            attObj.put("attributeName",att);
            attObj.put("type",type);
            attObj.put("data",dataList);
            attributes.add(attObj);
        }
        JSONObject response = new JSONObject();
        response.put("attributes", attributes);
        return response.toJSONString();
    }

    @RequestMapping(value = "/get_gbn", method = RequestMethod.POST)
    public String get_gbn(HttpServletRequest request){
        String method = request.getParameter("method");
        if(method != null){
            return this.bn.getGlobalGBN(method);
        } else{
            return this.bn.getGlobalGBN();
        }
    }

    @RequestMapping(value = "/get_recommendation", method = RequestMethod.POST)
    public String get_local_gbn(HttpServletRequest request) {
        List<String> selectAtt = JSON.parseArray(request.getParameter("attributes"), String.class);
        if(selectAtt != null){
            return this.bn.getRecommendation(selectAtt);
        }else{
            return this.bn.getRecommendation();
        }
    }
}
