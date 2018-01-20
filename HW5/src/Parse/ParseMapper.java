package page_rank;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ParseMapper extends Mapper<LongWritable, Text, Text, Text> {

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        context.getCounter("Nodes", "N").increment(1);
        /*  Match title pattern */
        Pattern titlePattern = Pattern.compile("<title>(.+?)</title>");
        Matcher titleMatcher = titlePattern.matcher(value.toString());
        titleMatcher.find();
        // No need capitalizeFirstLetter
        Text title = new Text(unescapeXML(titleMatcher.group(1)));
        // <title, "#">
        context.write(title, new Text("#"));

        /*  Match link pattern */
        Pattern linkPattern = Pattern.compile("\\[\\[(.+?)([\\|#]|\\]\\])");
        Matcher linkMatcher = linkPattern.matcher(value.toString());
        while (linkMatcher.find()) {
            // Need capitalizeFirstLetter
            String link = unescapeXML(capitalizeFirstLetter(linkMatcher.group(1)));
            // <link, title>
            context.write(new Text(link), title);
        }
    }

    private String unescapeXML(String input) {
        return input
            .replaceAll("&lt;", "<")
            .replaceAll("&gt;", ">")
            .replaceAll("&amp;", "&")
            .replaceAll("&quot;", "\"")
            .replaceAll("&apos;", "\'");
    }

    private String capitalizeFirstLetter(String input) {
        char firstChar = input.charAt(0);

        if (firstChar >= 'a' && firstChar <= 'z') {
            if (input.length() == 1)
                return input.toUpperCase();
            else
                return input.substring(0, 1).toUpperCase() + input.substring(1);
        }
        return input;
    }
}

