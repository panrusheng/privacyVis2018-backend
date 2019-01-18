package tk.mybatis.springboot.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.filereaders.DataFileReader;
import eu.amidst.core.datastream.filereaders.DataRow;
import eu.amidst.core.datastream.filereaders.arffFileReader.DataRowWeka;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.core.variables.stateSpaceTypes.SparseFiniteStateSpace;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This class implements the interface {@link DataFileReader} and defines an ARFF (Weka Attribute-Relation File Format) data reader.
 */
public class MyArffReader implements DataFileReader {

    private static int category_cnt = 2;

    private static int max_atts_num = 20;

    public final double[] min = new double[max_atts_num];

    public final double[] max = new double[max_atts_num];

    /** Represents the relation name. */
    String relationName;

    /** Represents the list of {@link Attributes}. */
    private Attributes attributes;

    /** Represents the data line count. */
    private int dataLineCount;

    /** Represents the path of the ARFF file to be read. */
    private Path pathFile;

    /** Represents an array of {@link StateSpaceTypeEnum} for the corresponding list of {@link Attributes}. */
    private StateSpaceTypeEnum[] stateSpace;

    /** Represents a {@code Stream} of {@code DataRow}. */
    private Stream<DataRow> streamString;

    /**
     * Creates an {@link Attribute} from a given index and line.
     * @param index an {@code int} that represents the index of column to which the Attribute refers.
     * @param line a {@code String} starting with "@attribute" and including the name of the Attribute and its state space type.
     * @return an {@link Attribute} object.
     */
    public static Attribute createAttributeFromLine(int index, String line){
        String[] parts = line.split("\\s+|\t+");

        if (!parts[0].trim().startsWith("@attribute"))
            throw new IllegalArgumentException("Attribute line does not start with @attribute");

        String name = parts[1].trim();
        //name = StringUtils.strip(name,"'");

        name = name.replaceAll("^'+", "");
        name = name.replaceAll("'+$", "");

        parts[2]=parts[2].trim();

        if (parts[2].equals("real") || parts[2].equals("numeric")){
            String[] states = new String[category_cnt];
            for(int i = 0; i < category_cnt; i++){
                states[i] = "category_" + i;
            }
            List<String> statesNames = Arrays.stream(states).map(String::trim).collect(Collectors.toList());
            return new Attribute(index, name, new FiniteStateSpace(statesNames));
        }else if (parts[2].startsWith("{")){
            parts[2]=line.substring(line.indexOf("{")).replaceAll("\t", "");
            String[] states = parts[2].substring(1,parts[2].length()-1).split(",");

            List<String> statesNames = Arrays.stream(states).map(String::trim).collect(Collectors.toList());

            return new Attribute(index, name, new FiniteStateSpace(statesNames));
        }else if (parts[2].equals("SparseMultinomial")) {
            return new Attribute(index, name, new SparseFiniteStateSpace(Integer.parseInt(parts[3])));
        }else{
            throw new UnsupportedOperationException("We can not create an attribute from this line: "+line);
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void loadFromFile(String pathString) {
        pathFile = Paths.get(pathString);
        Supplier<Stream<String>> streamSupplier = () -> {
            try{
                return Files.lines(pathFile)
                        .map(String::trim)
                        .filter(w -> !w.isEmpty())
                        .filter(w -> !w.startsWith("%"));
            }catch (IOException ex){
                throw new UncheckedIOException(ex);
            }};

        Optional<String> atRelation = streamSupplier.get()
                .limit(1)
                .filter(line -> line.startsWith("@relation"))
                .findFirst();

        if (!atRelation.isPresent())
            throw new IllegalArgumentException("ARFF file does not start with a @relation line.");

        relationName = atRelation.get().split(" ")[1];

        final int[] count = {0};
        Optional<String> atData = streamSupplier.get()
                .peek(line -> count[0]++)
                .filter(line -> line.startsWith("@data"))
                .findFirst();

        if (!atData.isPresent())
            throw new IllegalArgumentException("ARFF file does not contain @data line.");

        dataLineCount = count[0];

        List<String> attLines = streamSupplier.get()
                .limit(dataLineCount)
                .filter(line -> line.startsWith("@attribute"))
                .collect(Collectors.toList());

        List<Attribute> atts = IntStream.range(0,attLines.size())
                .mapToObj( i -> createAttributeFromLine(i, attLines.get(i)))
                .collect(Collectors.toList());

        this.attributes = new Attributes(atts);

        //
        stateSpace=new StateSpaceTypeEnum[atts.size()];

        for (Attribute att: atts){
            stateSpace[att.getIndex()] = att.getStateSpaceType().getStateSpaceTypeEnum();
        }

        Optional<String> firstRecord = streamSupplier.get()
                .skip(this.dataLineCount)
                .filter(w -> !w.isEmpty())
                .findFirst();
        int numberOfAttributes = this.attributes.getNumberOfAttributes();
        if(firstRecord.isPresent()) {
            for(int i = 0, len_i = numberOfAttributes; i < len_i; i++){
                if(this.attributes.getFullListOfAttributes().get(i).stringValue(0.0).equals("category_0")) {
                    final int ii = i;
                    double firstRecordVal = Double.valueOf(firstRecord.get().split(",")[i]);
                    min[i] = firstRecordVal;
                    max[i] = firstRecordVal;
                    streamSupplier.get()
                            .skip(this.dataLineCount)
                            .filter(w -> !w.isEmpty())
                            .forEach(w -> {
                                Double recordVal = Double.valueOf(w.split(",")[ii]);
                                if (recordVal < min[0]) {
                                    min[ii] = recordVal;
                                } else if (recordVal > max[0]) {
                                    max[ii] = recordVal;
                                }
                            });
                }
            }
        }
    }

     /**
     * {@inheritDoc}
     */
    @Override
    public Attributes getAttributes() {
        return this.attributes;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean doesItReadThisFile(String fileName) {
        if (new File(fileName).isDirectory())
            return false;
        String[] parts = fileName.split("\\.");
        return parts[parts.length-1].equals("arff");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Stream<DataRow> stream() {
        try {
            streamString = Files.lines(pathFile)
                    .filter(w -> !w.isEmpty())
                    .filter(w -> !w.startsWith("%"))
                    .skip(this.dataLineCount)
                    .filter(w -> !w.isEmpty())
                    .map(line -> new MyDataRowWeka(this.attributes, line, min, max));
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        return streamString;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void restart(){
        streamString = null;
    }

}

