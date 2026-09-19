-- Keep headings readable and allow long repository paths/hashes to wrap.
function Header(h)
  if FORMAT:match('latex') and h.identifier == 'references' then
    return {pandoc.RawBlock('latex', '\\Needspace{10\\baselineskip}'), h,
            pandoc.RawBlock('latex', '\\small')}
  end
end
function Code(c)
  if FORMAT:match('latex') and #c.text > 28 and not c.text:find('[{}\\]') then
    return pandoc.RawInline('latex', '\\nolinkurl{' .. c.text .. '}')
  end
end
