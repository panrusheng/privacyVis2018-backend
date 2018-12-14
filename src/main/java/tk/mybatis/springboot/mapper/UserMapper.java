package tk.mybatis.springboot.mapper;

import org.apache.ibatis.annotations.Select;
import tk.mybatis.springboot.model.User;
import tk.mybatis.springboot.util.MyMapper;

import java.util.List;

/**
 * @author liuzh_3nofxnp
 * @since 2016-01-22 22:17
 */
public interface UserMapper extends MyMapper<User> {
}
