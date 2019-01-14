package tk.mybatis.springboot.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.github.pagehelper.PageInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import tk.mybatis.springboot.model.User;
import tk.mybatis.springboot.service.UserService;

import java.util.Arrays;
import java.util.List;

import tk.mybatis.springboot.util.algorithm.Bayes;

import javax.servlet.http.HttpServletRequest;

@RestController
@RequestMapping("/api")
public class UserController {

    @Autowired
    private UserService UserService;

    private JSONObject attrList;

    UserController() {
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
        this.attrList.put("FE" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a further education within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("HE" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a higher education within six years?\", \"type\": \"numerical\"}"));
        this.attrList.put("ascc" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep at school within six years?\", \"type\": \"numerical\"}"));
    }

    @RequestMapping//Home
    public List<User> getAll() {
        return UserService.getAll();
    }

    @RequestMapping(value = "/load_data", method = RequestMethod.POST)
    public String loadData(HttpServletRequest request){
//        String datasetName = request.getParameter("dataset");
//        if(datasetName.equals("user")){}
        JSONObject response = new JSONObject();
        response.put("attrList",this.attrList);
        return response.toJSONString();
    }

    @RequestMapping(value = "/get_attribute_distribution", method = RequestMethod.POST)
    public String get_attribute_distribution(HttpServletRequest request) {
        List<User> userList = UserService.getAll();
        List<String> selectAttr = JSON.parseArray(request.getParameter("attributes"), String.class);

        JSONArray attributes = new JSONArray();
        for(String attr: selectAttr){
            JSONObject attrObj = new JSONObject();
            String type = (String)((JSONObject) this.attrList.get(attr)).get("type");
            if(type.equals("categorical")){//type = categorical
                List<Category> dataList;
                for(User _user: userList){
                    JSONObject user = JSONObject.parseObject(JSONObject.toJSONString(_user));
                    String attribute = (String)user.get(attr);

                }
            }
            else{//type = numerical

            }
            attrObj.put("attributeName",attr);
            attrObj.put("type",type);
//            attrObj.put("data",JSON.parseObject(dataList.toString()));
            attributes.add(attrObj);
        }
        JSONObject response = new JSONObject();
        response.put("attributes", attributes);
        return response.toJSONString();
    }

    @RequestMapping(value = "/add")
    public User add() {
        return new User();
    }

    @RequestMapping(value = "/view/{id}")
    public User view(@PathVariable Integer id) {
        ModelAndView result = new ModelAndView();
        User User = UserService.getById(id);
        return User;
    }

    @RequestMapping(value = "/get_gbn")
    public String get_gbn(){
        return Bayes.getGBN();
    }

}

class Category {

    private String category;

    private float value;

    Category(){
        this.category = "empty";
        this.value = 0.5f;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getCategory() {
        return this.category;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public float getValue() {
        return this.value;
    }
}

class Number {

    private List<Float> label;//2-tuple

    private float value;

    Number(){
        this.label = Arrays.asList(0.0f, 1.0f);
        this.value = 0.5f;
    }

    public void setLabel(List<Float> label) {
        this.label = label;
    }

    public List<Float> getLabel() {
        return this.label;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public float getValue() {
        return this.value;
    }
}
