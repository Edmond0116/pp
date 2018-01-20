package page_rank;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.Text;

public class SortPair implements WritableComparable {
    private Text key;
    private DoubleWritable val;

    public SortPair() {
        key = new Text();
        val = new DoubleWritable();
    }

    public SortPair(Text key, DoubleWritable value) {
        this.key = key;
        this.val = value;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        key.write(out);
        val.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        key.readFields(in);
        val.readFields(in);
    }

    public Text getKey() {
        return key;
    }

    public DoubleWritable getVal() {
        return val;
    }

    @Override
    public int compareTo(Object o) {
        SortPair rhs = (SortPair) o;
        int res = val.compareTo(rhs.getVal());
        if (res != 0) return -res;
        return key.toString().compareTo(rhs.getKey().toString());
    }
} 

