import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import edu.stanford.nlp.scenegraph.SceneGraphNode;
import edu.stanford.nlp.scenegraph.SceneGraphAttribute;
import edu.stanford.nlp.scenegraph.SceneGraphRelation;
import java.util.Iterator;
import java.util.List;

public class CocoCaptionParser {
    public static void main(String[] args) throws IOException, ParseException {
        RuleBasedParser parser = new RuleBasedParser();

        // input captions
        String coco_org_caption_filename = "data\\COCO2017\\annotations\\captions_train2017.json";
        String out_filename = "data\\COCO2017\\annotations\\captions_train2017_with_scene_graph.json";
        OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(out_filename),"UTF-8");

        // read file
        JSONParser jparser = new JSONParser();
        JSONObject caption_annotations = (JSONObject) jparser.parse(new FileReader(coco_org_caption_filename));
        JSONArray captions = (JSONArray) caption_annotations.get("annotations");

        int count = 0;
        for (Object o : captions) {
            JSONObject cap = (JSONObject) o;

            String sentence = (String) cap.get("caption");
            SceneGraph sg = parser.parse(sentence);
//            System.out.println(sg.toReadableString());

            cap.put("text_scene_graph", toJSON(sg, sentence));

            count += 1;
            if ((count % 100) == 0) {
                System.out.println(String.format("%d", count) + "/" + String.format("%d", captions.size()));
//                break;
            }
        }

        osw.write(caption_annotations.toString());
        osw.flush();
        osw.close();
        System.out.println("Finish!");
    }

    public static String toJSON(SceneGraph sg, String sent) {
        JSONObject obj = new JSONObject();
        obj.put("sentence", sent);
        List<SceneGraphNode> objects = sg.nodeListSorted();
        JSONArray attrs = new JSONArray();
        Iterator var7 = objects.iterator();

        Iterator var9;
        JSONObject objObj;
        JSONArray names;
        while(var7.hasNext()) {
            SceneGraphNode node = (SceneGraphNode)var7.next();
            var9 = node.getAttributes().iterator();

            while(var9.hasNext()) {
                SceneGraphAttribute attr = (SceneGraphAttribute)var9.next();
                objObj = new JSONObject();
                objObj.put("attribute", attr.toString());
                objObj.put("object", attr.toString());
                objObj.put("predicate", "is");
                objObj.put("subject", objects.indexOf(node));
                names = new JSONArray();
                names.add(node.toJSONString());
                names.add("is");
                names.add(attr.toString());
                objObj.put("text", names);
                attrs.add(objObj);
            }
        }

        obj.put("attributes", attrs);
        JSONArray relns = new JSONArray();
        Iterator var14 = sg.relationListSorted().iterator();

        while(var14.hasNext()) {
            SceneGraphRelation reln = (SceneGraphRelation)var14.next();
            JSONObject relnObj = new JSONObject();
            relnObj.put("predicate", reln.getRelation());
            relnObj.put("subject", objects.indexOf(reln.getSource()));
            relnObj.put("object", objects.indexOf(reln.getTarget()));
            JSONArray text = new JSONArray();
            text.add(reln.getSource().toJSONString());
            text.add(reln.getRelation());
            text.add(reln.getTarget().toJSONString());
            relnObj.put("text", text);
            relns.add(relnObj);
        }

        obj.put("relationships", relns);
        JSONArray objs = new JSONArray();
        var9 = objects.iterator();

        while(var9.hasNext()) {
            SceneGraphNode node = (SceneGraphNode)var9.next();
            objObj = new JSONObject();
            names = new JSONArray();
            names.add(node.toJSONString());
            objObj.put("names", names);

            // extra infos
            JSONArray pos = new JSONArray();
            JSONArray char_span = new JSONArray();
            pos.add(node.value().beginPosition());
            pos.add(node.value().endPosition());
            char_span.add(pos);
            objObj.put("char_span", char_span);

            JSONArray lemma = new JSONArray();
            lemma.add(node.value().lemma());
            objObj.put("lemma", lemma);

            objs.add(objObj);
        }

        obj.put("objects", objs);
        return obj.toJSONString();
    }
}

