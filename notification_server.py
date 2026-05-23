import json
import ctypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

class NotificationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if data.get('type') == 'training_complete':
                pattern = data.get('pattern', 'N/A')
                img_auc = data.get('img_auc', 'N/A')
                pixel_auc = data.get('pixel_auc', 'N/A')
                message = f"训练完成！\n\n数据集编号: {pattern}\n图像级AUC: {img_auc}\n像素级AUC: {pixel_auc}"
                title = "训练通知"
            elif data.get('type') == 'all_complete':
                message = f"所有训练任务已完成！\n\n总任务数: {data.get('total', 0)}\n成功: {data.get('completed', 0)}\n失败: {data.get('failed', 0)}"
                title = "训练完成通知"
            else:
                message = "收到未知通知"
                title = "通知"
            
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1 | 0x1000)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success'}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))
    
    def log_message(self, format, *args):
        pass

def run_server(port=9999):
    server_address = ('', port)
    httpd = HTTPServer(server_address, NotificationHandler)
    print(f"通知服务已启动，监听端口: {port}")
    print(f"服务地址: http://localhost:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        httpd.server_close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练通知服务')
    parser.add_argument('--port', type=int, default=9999, help='监听端口')
    args = parser.parse_args()
    run_server(args.port)
