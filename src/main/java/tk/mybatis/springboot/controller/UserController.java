package tk.mybatis.springboot.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import tk.mybatis.springboot.model.User;
import tk.mybatis.springboot.service.UserService;

import java.util.List;

import tk.mybatis.springboot.util.Bayes;
import tk.mybatis.springboot.util.MyMath;

import javax.servlet.http.HttpServletRequest;

@RestController
@RequestMapping("/api")
public class UserController {

    @Autowired
    private UserService UserService;

    private JSONObject attrList;

    private void initAttrList(){
        this.attrList = new JSONObject();
        this.attrList.put("wei" ,  JSON.parseObject("{\"description\" : \"Data adjustment factor\", \"type\": \"numerical\"}"));
        this.attrList.put("gen" ,  JSON.parseObject("{\"description\" : \"gender\", \"type\": \"categorical\"}"));
        this.attrList.put("cat" ,  JSON.parseObject("{\"description\" : \"Whether he or she is a Catholic believer?\", \"type\": \"categorical\"}"));
        this.attrList.put("res" ,  JSON.parseObject("{\"description\" : \"Residence\", \"type\": \"categorical\"}"));
        this.attrList.put("sch" ,  JSON.parseObject("{\"description\" : \"School\", \"type\": \"categorical\"}"));
        this.attrList.put("fue" ,  JSON.parseObject("{\"description\" : \"Whether the father is unemployed?\", \"type\": \"categorical\"}"));
        this.attrList.put("gcs" ,  JSON.parseObject("{\"description\" : \"Whether he or she has five or more GCSEs at grades AC?\", \"type\": \"categorical\"}"));
        this.attrList.put("fmp" ,  JSON.parseObject("{\"description\" : \"Whether the father is at least the management?\", \"type\": \"categorical\"}"));
        this.attrList.put("lvb" ,  JSON.parseObject("{\"description\" : \"Whether live with parents?\", \"type\": \"categorical\"}"));
        this.attrList.put("tra" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep training within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("emp" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep employment within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("jol" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep joblessness within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("fe" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a further education within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("he" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a higher education within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("ascc" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep at school within six years?\", \"type\": \"numerical\"}"));
    }

    UserController() {
        initAttrList();
    }

    @RequestMapping//Home
    public List<User> getAll() {
        return UserService.getAll();
    }

    @RequestMapping(value = "/load_data", method = RequestMethod.POST)
    public String loadData(HttpServletRequest request){
        JSONObject response = new JSONObject();
        response.put("attrList",this.attrList);
        return response.toJSONString();
    }

    @RequestMapping(value = "/get_attribute_distribution", method = RequestMethod.POST)
    public String get_attribute_distribution(HttpServletRequest request) {
        List<String> selectAtt = JSON.parseArray(request.getParameter("attributes"), String.class);
        JSONArray attributes = new JSONArray();
        Bayes bn = new Bayes();
        for(String att: selectAtt){
            JSONObject attrObj = new JSONObject();
            String type = (String)((JSONObject) this.attrList.get(att)).get("type");
            JSONArray dataList = bn.getAttDistribution(att, type);
            attrObj.put("attributeName",att);
            attrObj.put("type",type);
            attrObj.put("data",dataList);
            attributes.add(attrObj);
        }
        JSONObject response = new JSONObject();
        response.put("attributes", attributes);
        return response.toJSONString();
    }

    @RequestMapping(value = "/get_gbn", method = RequestMethod.POST)
    public String get_gbn(HttpServletRequest request){
        String method = request.getParameter("method");
        Bayes bn = new Bayes();
        if(method != null){
            return bn.getGlobalGBN(method);
        } else{
            return bn.getGlobalGBN();
        }
    }

    @RequestMapping(value = "/recommendation", method = RequestMethod.POST)
    public String get_local_gbn(HttpServletRequest request) {
        Bayes bn = new Bayes();
        return bn.recommendGroup();
    }
}
