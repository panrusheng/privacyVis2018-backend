/*
 *
 *
 *    Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
 *    See the NOTICE file distributed with this work for additional information regarding copyright ownership.
 *    The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use
 *    this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under the License is
 *    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under the License.
 *
 *
 */

package tk.mybatis.springboot.util;

/**
 * Created by andresmasegosa on 1/9/15.
 */

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.filereaders.DataRow;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import org.w3c.dom.Attr;

/**
 * This class implements the interface {@link DataRow} and defines a Weka data row.
 */
public class MyDataRowWeka implements DataRow{

    /** Represents an {@code array} of double. */
    double[] data;

    /** Represents the list of {@link Attributes}. */
    Attributes atts;

    /**
     * Creates a new DataRowWeka from a given line and list of attributes.
     * @param atts_ an input list of the list of {@link Attributes}.
     * @param line a {@code String} including the values of the corresponding input attributes.
     */
    public MyDataRowWeka(Attributes atts_, String line, double[] min, double[] max){
        atts = atts_;
        data = new double[atts.getNumberOfAttributes()];
        String[] parts = line.split(",");
        for (int i = 0; i < parts.length; i++) {
            parts[i] = parts[i].trim();
        }
        if (parts.length!=atts.getNumberOfAttributes()) {
            throw new IllegalStateException("The number of columns does not match the number of attributes.");
        }
        for (int i = 0; i < parts.length; i++) {
            if(parts[i].equals("?")){
                data[i] = Double.NaN;
            }
            else {
                Attribute att = atts.getFullListOfAttributes().get(i);
                switch (att.getStateSpaceType().getStateSpaceTypeEnum()) {
                    case REAL:
                        try{
                            data[i] = Double.parseDouble(parts[i]);
                        }catch(Exception ex){
                            System.out.println("Error Reading ARFF:");
                            System.out.println("Attribute Name: " + att.getName());
                            System.out.println("Error when reading value: " + parts[i]);

                            ex.printStackTrace();
                            new Exception();
                        }
                        break;
                    case FINITE_SET:
                        try {
                            if(att.stringValue(0.0).equals("category_0")){
                                if(Double.valueOf(parts[i]) < ( min[i] +max[i] ) / 2 ){
                                    data[i] = 0.0;
                                }
                                else {
                                    data[i] = 1.0;
                                }
                            }else{
                                FiniteStateSpace finiteStateSpace = att.getStateSpaceType();
                                data[i] = finiteStateSpace.getIndexOfState(parts[i]);
                            }
                        }catch(Exception ex){
                            System.out.println("Error Reading ARFF:");
                            System.out.println("Attribute Name: " + att.getName());
                            System.out.print("Attribute States: ");
                            ((FiniteStateSpace)att.getStateSpaceType()).getStatesNames().stream().forEach(state -> System.out.print(state + ", "));
                            System.out.println();
                            System.out.println("Error when reading value: " + parts[i]);

                            new Exception();
                        }
                        break;
                    case SPARSE_FINITE_SET:
                        data[i] = Integer.parseInt(parts[i]);
                }
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getValue(Attribute att) {
        return data[att.getIndex()];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setValue(Attribute att, double value) {
        this.data[att.getIndex()]=value;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Attributes getAttributes() {
        return atts;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] toArray() {
        return data;
    }


}
