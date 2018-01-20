package page_rank;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class ParseReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        List<String> titles = new ArrayList<String>();
        for (Text val: values)
            titles.add(val.toString());
        if (titles.contains("#")) {
            for (String title : titles) {
                if (title.equals("#"))
                    context.write(key, new Text("#"));
                else
                    context.write(new Text(title), key);
            }
        }
    }
}
