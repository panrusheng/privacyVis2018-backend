package tk.mybatis.springboot.controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import tk.mybatis.springboot.model.User;
import tk.mybatis.springboot.service.UserService;

import java.io.BufferedReader;
import java.util.List;

import tk.mybatis.springboot.util.algorithm.Bayes;

import javax.servlet.http.HttpServletRequest;

@CrossOrigin
@RestController
@RequestMapping("/api")
public class UserController {

    @Autowired
    private UserService UserService;

    private Bayes bn;

    UserController() {
        bn = new Bayes("user");
    }

    @RequestMapping//Home
    public List<User> getAll() {
        return UserService.getAll();
    }

    @RequestMapping(value = "load_data", method = RequestMethod.POST)
    public String loadData(HttpServletRequest request){
        String dataset = request.getParameter("dataset");

        if (dataset != null) {
            bn = new Bayes(dataset);
        }
        
        JSONObject response = new JSONObject();
        response.put("attList",bn.getAttDescription());
        return response.toJSONString();
    }

    @RequestMapping(value = "get_gbn", method = RequestMethod.POST)
    public String get_gbn(HttpServletRequest request){
        List<JSONObject> selectAtt = JSON.parseArray(request.getParameter("attributes"), JSONObject.class);
        String riskLimit = request.getParameter("riskLimit");
        String method = request.getParameter("method");

        this.bn.setRiskLimit(Double.valueOf(riskLimit));
        if(method == null) {
            return this.bn.getGBN(selectAtt);
        }
        else{
            return this.bn.getGBN(method, selectAtt);
        }
    }

    @RequestMapping(value = "edit_gbn", method = RequestMethod.POST)
    public String edit_gbn(HttpServletRequest request){
        StringBuilder buffer = new StringBuilder();
        String line;
        try {
            BufferedReader reader = request.getReader();
            while ((line = reader.readLine()) != null) {
                buffer.append(line);
            }
        } catch (Exception e) {

        }
        JSONObject parameters = JSON.parseObject(buffer.toString());
        List<JSONObject> events = JSON.parseArray(parameters.getString("events"), JSONObject.class);
        return bn.editGBN(events);
    }

    @RequestMapping(value = "get_recommendation", method = RequestMethod.POST)
    public String get_recommendation(HttpServletRequest request) {
        StringBuilder buffer = new StringBuilder();
        String line;
        try {
            BufferedReader reader = request.getReader();
            while ((line = reader.readLine()) != null) {
                buffer.append(line);
            }
        } catch (Exception e) {

        }

        JSONObject parameters = JSON.parseObject(buffer.toString());
        List<JSONObject> links = JSON.parseArray(parameters.getString("links"), JSONObject.class);
        List<JSONObject> utilityList = JSON.parseArray(parameters.getString("utilityList"), JSONObject.class);
        return this.bn.getRecommendation(links, utilityList);
    }

    @RequestMapping(value = "get_result", method = RequestMethod.POST)
    public String get_result(HttpServletRequest request) {
        StringBuilder buffer = new StringBuilder();
        String line;
        try {
            BufferedReader reader = request.getReader();
            while ((line = reader.readLine()) != null) {
                buffer.append(line);
            }
        } catch (Exception e) {

        }
        JSONObject parameters = JSON.parseObject(buffer.toString());
        List<JSONObject> options = JSON.parseArray(parameters.getString("options"), JSONObject.class);
        return this.bn.getResult(options);
    }

    @RequestMapping(value = "get_test", method = RequestMethod.POST)
    public String get_test(HttpServletRequest request) {
        String model = request.getParameter("method");

        JSONObject options = JSON.parseObject(request.getParameter("options"));
        List<String> trimList = JSONObject.parseArray(request.getParameter("trimList")).toJavaList(String.class);
        return bn.getTest(model, options, trimList).toJSONString();
    }
}
