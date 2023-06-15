import { Configuration, CreateEmbeddingResponse, OpenAIApi } from "npm:openai";
import * as similarity from "https://esm.sh/compute-cosine-similarity";

const configuration = new Configuration({
  apiKey: Deno.env.get("API_KEY"),
});
const openai = new OpenAIApi(configuration);

async function getEmbedding(input: string): Promise<number[]> {
  const resp = await openai.createEmbedding({
    input: input,
    model: "text-embedding-ada-002",
  });

  return resp.data.data[0].embedding;
}

async function main() {
  const vec1 = await getEmbedding(
    "カレンダーから洋服の登録ができない。服の読み込み画面で登録した洋服が読み込み中のまま更新されない",
  );
  const vec2 = await getEmbedding(
    "何度もエラーになりましたのメッセージが出て自分のカレンダーコーデの登録や回答の記入ができないことがあります\n問合せもなかなかできなかったです",
  );
  const vec3 = await getEmbedding(
    "家族分も別のクローゼットで管理できるようにしてほしい",
  );

  console.log("Vec1 - Vec2 Similarity -> ", similarity.default(vec1, vec2));
  console.log("Vec1 - Vec3 Similarity -> ", similarity.default(vec1, vec3));
  console.log("Vec2 - Vec3 Similarity -> ", similarity.default(vec2, vec3));
}

main();
