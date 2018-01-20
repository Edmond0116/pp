package page_rank;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class SortGroupComparator extends WritableComparator {
    
    public SortGroupComparator() {
        super(SortPair.class, true);
    }

    public int compare(WritableComparable o1, WritableComparable o2) {
        SortPair key1 = (SortPair) o1, key2 = (SortPair) o2;
        return -key1.getVal().compareTo(key2.getVal());
    }
}
