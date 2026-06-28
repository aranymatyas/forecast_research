function Figure(fig)
  local img = fig.content[1].content[1]
  if not img or img.t ~= 'Image' then return fig end

  local caption_text = pandoc.utils.stringify(fig.caption.long[1])
  local dir = img.src:match("([^/]+)/[^/]+$")
  local name = img.src:match("([^/]+)%.%w+$")
  local label = ''
  if dir and name then
    label = '\\label{' .. dir .. '/' .. name .. '}'
  elseif name then
    label = '\\label{' .. name .. '}'
  end

  local latex = '\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{' .. img.src .. '}\n\\caption{' .. caption_text .. label .. '}\n\\end{figure}'
  return pandoc.RawBlock('latex', latex)
end
