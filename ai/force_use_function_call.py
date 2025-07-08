// deno-lint-ignore-file no-explicit-any
import {serve} from "https://deno.land/std@0.210.0/http/server.ts";
import { Status } from "https://deno.land/std@0.210.0/http/status.ts";

// --- 配置区域 ---
const CONFIG = {
  // 上游 OpenAI 兼容服务的地址
  UPSTREAM_BASE_URL: "https://api.groq.com/openai",
  // 用于访问上游服务的 API Key
  UPSTREAM_API_KEY: Deno.env.get("UPSTREAM_API_KEY") || "YOUR_UPSTREAM_API_KEY_HERE",
  // 允许访问此中间件的客户端 API Key 列表
  ALLOWED_CLIENT_KEYS: new Set([
    "sk-my-secret-key-1",
    "sk-my-secret-key-2",
  ]),
  // 服务器监听端口
  PORT: 8000,
};
// --- 配置区域结束 ---

/**
 * 这是注入给模型的系统提示模板，用于引导模型进行函数调用。
 * 它会根据客户端请求中定义的工具动态填充。
 */
const FUNCTION_CALL_PROMPT_TEMPLATE = `你可以使用以下工具来帮助你解决问题：

工具列表：

{TOOLS_LIST}

当你判断需要使用工具时，必须严格遵循以下格式：

1. 回答的第一行必须是：
FC_USE
没有任何前、尾随空格，全大写。

2. 然后，在回答的最后，请使用如下格式输出函数调用（使用 XML 语法）：

<function_call>
  <tool>tool_name</tool>
  <args>
    <key1>value1</key1>
    <key2>value2</key2>
  </args>
</function_call>

注意事项：
- 除非你确定需要调用工具，否则不要输出 FC_USE。
- 你只能调用一个工具。
- 保证输出的 XML 是有效的、严格符合上述格式。
- 不要随便更改格式。
- 你单回合只能调用一次工具。

现在请准备好遵循以上规范。`;

/**
 * 根据客户端请求中的 tools 定义，生成注入的系统提示。
 * @param tools - OpenAI 格式的工具数组
 * @returns 格式化后的系统提示字符串
 */
function generateFunctionPrompt(tools: any[]): string {
  const toolsList = tools.map((tool) => {
    const func = tool.function;
    const params = Object.entries(func.parameters?.properties ?? {})
      .map(([name, prop]: [string, any]) => `${name} (${prop.type})`)
      .join(", ");
    return `${tools.indexOf(tool) + 1}. <tool name="${func.name}" description="${func.description}">\n   参数：${params || "无"}`;
  }).join("\n\n");

  return FUNCTION_CALL_PROMPT_TEMPLATE.replace("{TOOLS_LIST}", toolsList);
}

/**
 * 解析模型输出的 Function Call XML。
 * @param xmlString - 包含 <function_call> 的 XML 字符串
 * @returns 解析后的工具名和参数对象，或 null
 */
function parseFunctionCallXml(xmlString: string): { name: string; args: Record<string, string> } | null {
  const toolMatch = /<tool>(.*?)<\/tool>/.exec(xmlString);
  if (!toolMatch) return null;
  const name = toolMatch[1].trim();

  const args: Record<string, string> = {};
  const argsBlockMatch = /<args>([\s\S]*?)<\/args>/.exec(xmlString);
  if (argsBlockMatch) {
    const argsContent = argsBlockMatch[1];
    const argRegex = /<(\w+)>(.*?)<\/(\w+)>/g;
    let match;
    while ((match = argRegex.exec(argsContent)) !== null) {
      if (match[1] === match[3]) {
        args[match[1]] = match[2];
      }
    }
  }

  return { name, args };
}


/**
 * 主请求处理器
 * @param request - Deno.serve 传入的请求对象
 * @returns Response 对象
 */
async function handler(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const authHeader = request.headers.get("Authorization");
  const clientKey = authHeader?.replace("Bearer ", "");

  if (!clientKey || !CONFIG.ALLOWED_CLIENT_KEYS.has(clientKey)) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: Status.Unauthorized,
      headers: { "Content-Type": "application/json" },
    });
  }

  const upstreamUrl = new URL(CONFIG.UPSTREAM_BASE_URL);
  upstreamUrl.pathname = url.pathname;
  upstreamUrl.search = url.search;

  // 路由: /v1/models
  if (url.pathname.endsWith("/v1/models")) {
    const upstreamRequest = new Request(upstreamUrl, {
      method: request.method,
      headers: {
        "Authorization": `Bearer ${CONFIG.UPSTREAM_API_KEY}`,
      },
    });
    return fetch(upstreamRequest);
  }

  // 路由: /v1/chat/completions
  if (url.pathname.endsWith("/v1/chat/completions")) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: Status.MethodNotAllowed });
    }

    const body = await request.json();
    let hasFunctionCall = false;

    if (body.tools && Array.isArray(body.tools) && body.tools.length > 0) {
      hasFunctionCall = true;
      const functionPrompt = generateFunctionPrompt(body.tools);
      
      const systemMessage = {
        role: "system",
        content: functionPrompt,
      };

      // 插入在最前方，尾随2换行以增强分隔效果
      body.messages.unshift(systemMessage);
      
      // 删除上游不兼容的字段
      delete body.tools;
      delete body.tool_choice;
    }
    
    const upstreamRequest = new Request(upstreamUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${CONFIG.UPSTREAM_API_KEY}`,
        "Accept": "text/event-stream",
      },
      body: JSON.stringify(body),
    });

    const upstreamResponse = await fetch(upstreamRequest);

    if (!upstreamResponse.ok) {
        return new Response(upstreamResponse.body, {
            status: upstreamResponse.status,
            statusText: upstreamResponse.statusText,
            headers: upstreamResponse.headers,
        });
    }

    // 如果没有函数调用，或者响应不是流式，直接代理
    if (!hasFunctionCall || !body.stream) {
      const responseText = await upstreamResponse.text();
      try {
        const responseJson = JSON.parse(responseText);
        if (hasFunctionCall && responseJson.choices?.[0]?.message?.content?.startsWith("FC_USE")) {
            const content = responseJson.choices[0].message.content;
            const parsedTool = parseFunctionCallXml(content);

            if (parsedTool) {
                const toolCallId = `call_${crypto.randomUUID().replace(/-/g, "")}`;
                responseJson.choices[0].message = {
                    role: "assistant",
                    content: null,
                    tool_calls: [{
                        id: toolCallId,
                        type: "function",
                        function: {
                            name: parsedTool.name,
                            arguments: JSON.stringify(parsedTool.args),
                        },
                    }],
                };
                responseJson.choices[0].finish_reason = "tool_calls";
            }
        }
        return new Response(JSON.stringify(responseJson), {
            status: 200,
            headers: { "Content-Type": "application/json" },
        });
      } catch (e) {
        // 如果解析失败，可能是上游错误，直接返回原文
        console.error("Error parsing upstream non-stream response:", e);
        return new Response(responseText, { status: 200, headers: upstreamResponse.headers });
      }
    }
    
    // 处理流式响应的函数调用转换
    const transformStream = createFunctionCallTransformStream(body.model);
    const responseStream = upstreamResponse.body!.pipeThrough(transformStream);
    
    return new Response(responseStream, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  }

  return new Response("Not Found", { status: Status.NotFound });
}

/**
 * 创建一个转换流，用于处理模型输出并转换函数调用格式。
 * @param model - 本次请求使用的模型名称
 * @returns TransformStream
 */
function createFunctionCallTransformStream(model: string) {
    let buffer = "";
    const prefix = "FC_USE\n";
    let state: "detecting" | "parsing_tool" | "passthrough" = "detecting";
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
  
    return new TransformStream({
      transform(chunk, controller) {
        const text = decoder.decode(chunk, { stream: true });
  
        if (state === "passthrough") {
          controller.enqueue(chunk);
          return;
        }
  
        buffer += text;

        if (state === "detecting") {
          if (buffer.startsWith(prefix)) {
            // 匹配到前缀，切换到工具解析模式
            state = "parsing_tool";
            // 从缓冲区移除前缀，剩下的内容是工具调用的开始
            buffer = buffer.substring(prefix.length);
            // 此处不 enqueue 任何东西，等待流结束时统一处理
          } else if (buffer.length >= prefix.length && !prefix.startsWith(buffer)) {
            // 前缀不匹配，切换到直通模式
            state = "passthrough";
            // 将已缓冲的所有内容发送出去
            controller.enqueue(encoder.encode(buffer));
            buffer = "";
          }
          // 如果 buffer 长度小于 prefix，则继续缓冲等待更多数据
        }
        // 如果状态是 parsing_tool，我们只是一直追加 buffer，直到流结束
      },
  
      flush(controller) {
        if (state === "parsing_tool") {
          // 流结束，此时 buffer 中是完整的工具调用 XML
          const parsedTool = parseFunctionCallXml(buffer);
          if (parsedTool) {
            const toolCallId = `call_${crypto.randomUUID().replace(/-/g, "")}`;
            const streamResponse = {
              id: `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`,
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: model,
              choices: [{
                index: 0,
                delta: {
                  role: "assistant",
                  content: null,
                  tool_calls: [{
                    index: 0,
                    id: toolCallId,
                    type: "function",
                    function: {
                      name: parsedTool.name,
                      arguments: JSON.stringify(parsedTool.args),
                    },
                  }],
                },
                finish_reason: "tool_calls",
              }],
            };
            // 按照 SSE 格式发送这个唯一的工具调用块
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(streamResponse)}\n\n`));
          } else {
            // 解析失败，可能模型输出格式错误，降级为普通文本输出
            controller.enqueue(encoder.encode(`data: {"id":"...","object":"chat.completion.chunk","created":${Math.floor(Date.now()/1000)},"model":"${model}","choices":[{"index":0,"delta":{"content":${JSON.stringify("Error: Failed to parse tool call. Raw output: " + buffer)}},"finish_reason":null}]}\n\n`));
          }
        } else if (buffer.length > 0) {
            // 如果在 detecting 状态下流就结束了，且 buffer 不为空（比如模型只输出了 "FC_"）
            // 把它作为普通内容发出去
            controller.enqueue(encoder.encode(buffer));
        }

        // 所有流都必须以 [DONE] 结尾
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      },
    });
  }

// --- 服务器启动 ---
console.log(`OpenAI Function Call Middleware starting...`);
console.log(`- Listening on: http://localhost:${CONFIG.PORT}`);
console.log(`- Proxying to: ${CONFIG.UPSTREAM_BASE_URL}`);
console.log(`- Allowed client keys: ${[...CONFIG.ALLOWED_CLIENT_KEYS].join(", ")}`);

serve(handler, { port: CONFIG.PORT });
