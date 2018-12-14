package tk.mybatis.springboot.model;

public class User extends BaseEntity{
    private Integer id;

    private Double wei;

    private Integer tra;

    private Integer emp;

    private Integer jol;

    private Integer fe;

    private Integer he;

    private Integer ascc;

    private String gen;

    private String cat;

    private String res;

    private String sch;

    private String fue;

    private String gcs;

    private String fmp;

    private String lvb;

    public User(Integer id, Double wei, Integer tra, Integer emp, Integer jol, Integer fe, Integer he, Integer ascc, String gen, String cat, String res, String sch, String fue, String gcs, String fmp, String lvb) {
        this.id = id;
        this.wei = wei;
        this.tra = tra;
        this.emp = emp;
        this.jol = jol;
        this.fe = fe;
        this.he = he;
        this.ascc = ascc;
        this.gen = gen;
        this.cat = cat;
        this.res = res;
        this.sch = sch;
        this.fue = fue;
        this.gcs = gcs;
        this.fmp = fmp;
        this.lvb = lvb;
    }

    public User() {
        super();
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Double getWei() {
        return wei;
    }

    public void setWei(Double wei) {
        this.wei = wei;
    }

    public Integer getTra() {
        return tra;
    }

    public void setTra(Integer tra) {
        this.tra = tra;
    }

    public Integer getEmp() {
        return emp;
    }

    public void setEmp(Integer emp) {
        this.emp = emp;
    }

    public Integer getJol() {
        return jol;
    }

    public void setJol(Integer jol) {
        this.jol = jol;
    }

    public Integer getFe() {
        return fe;
    }

    public void setFe(Integer fe) {
        this.fe = fe;
    }

    public Integer getHe() {
        return he;
    }

    public void setHe(Integer he) {
        this.he = he;
    }

    public Integer getascc() {
        return ascc;
    }

    public void setascc(Integer ascc) {
        this.ascc = ascc;
    }

    public String getGen() {
        return gen;
    }

    public void setGen(String gen) {
        this.gen = gen == null ? null : gen.trim();
    }

    public String getCat() {
        return cat;
    }

    public void setCat(String cat) {
        this.cat = cat == null ? null : cat.trim();
    }

    public String getRes() {
        return res;
    }

    public void setRes(String res) {
        this.res = res == null ? null : res.trim();
    }

    public String getSch() {
        return sch;
    }

    public void setSch(String sch) {
        this.sch = sch == null ? null : sch.trim();
    }

    public String getFue() {
        return fue;
    }

    public void setFue(String fue) {
        this.fue = fue == null ? null : fue.trim();
    }

    public String getGcs() {
        return gcs;
    }

    public void setGcs(String gcs) {
        this.gcs = gcs == null ? null : gcs.trim();
    }

    public String getFmp() {
        return fmp;
    }

    public void setFmp(String fmp) {
        this.fmp = fmp == null ? null : fmp.trim();
    }

    public String getLvb() {
        return lvb;
    }

    public void setLvb(String lvb) {
        this.lvb = lvb == null ? null : lvb.trim();
    }
}
