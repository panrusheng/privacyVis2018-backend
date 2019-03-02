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

    private Bayes bn;

    UserController() {
        bn = new Bayes();
    }

    @RequestMapping//Home
    public List<User> getAll() {
        return UserService.getAll();
    }

    @RequestMapping(value = "load_data", method = RequestMethod.POST)
    public String loadData(HttpServletRequest request){
        JSONObject response = new JSONObject();
        response.put("attList",bn.getAttDiscription());
        return response.toJSONString();
    }

    @RequestMapping(value = "get_gbn", method = RequestMethod.POST)
    public String get_gbn(HttpServletRequest request){
        List<JSONObject> selectAtt = JSON.parseArray(request.getParameter("attributes"), JSONObject.class);
        String method = request.getParameter("method");
        if(method == null) {
            return this.bn.getGBN(selectAtt);
        }
        else{
            return this.bn.getGBN(method, selectAtt);
        }
    }

    @RequestMapping(value = "edit_gbn", method = RequestMethod.POST)
    public String edit_gbn(HttpServletRequest request) {
        List<JSONObject> events = JSON.parseArray(request.getParameter("events"), JSONObject.class);
        //Todo
        return "";
    }

    @RequestMapping(value = "get_recommendation", method = RequestMethod.POST)
    public String get_local_gbn(HttpServletRequest request) {
        List<String> selectAtt = JSON.parseArray(request.getParameter("attributes"), String.class);
        return this.bn.getRecommendation(selectAtt);
    }

    @RequestMapping(value = "get_result", method = RequestMethod.POST)
    public String get_result(HttpServletRequest request) {
        List<JSONObject> options = JSON.parseArray(request.getParameter("options"), JSONObject.class);
        //Todo
        return "";
    }

    @RequestMapping(value = "set_trim", method = RequestMethod.POST)
    public String set_trim(HttpServletRequest request) {
        List<JSONObject> options = JSON.parseArray(request.getParameter("options"), JSONObject.class);
        //Todo
        return "";
    }

    @RequestMapping(value = "get_test", method = RequestMethod.POST)
    public String get_test(HttpServletRequest request) {
        //Todo
        return "";
    }
}
