export default function HtmlElement() {
  return (
    <iframe
      sandbox="allow-scripts allow-same-origin"
      srcDoc={props.html}             /* full file content arrives via prop */
      style={{ width: '100%', height: '500px', border: 'none' }}
    />
  );
}

