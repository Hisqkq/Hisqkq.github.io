module Jekyll
    class CustomTags < Liquid::Block
      def initialize(tag_name, markup, tokens)
        super
        # Check if the markup contains a space, indicating both tag type and title
        if markup.strip.include?(" ")
          @box_type, @title = markup.strip.split(" ", 2)
          @title = @title&.gsub(/^"(.*)"$/, '\1')
        else
          # If no space, treat the markup as only tag type and set title to nil
          @box_type = markup.strip
          @title = nil
        end
      end
  
      def render(context)
        content = super
        title_html = @title ? "<div class=\"title\">#{@title}</div>" : ""
        case @box_type
        when "info"
          "<div class=\"box-info\" markdown=\"1\">#{title_html}#{content}</div>"
        when "tip"
          "<div class=\"box-tip\" markdown=\"1\">#{title_html}#{content}</div>"
        when "warning"
          "<div class=\"box-warning\" markdown=\"1\">#{title_html}#{content}</div>"
        when "danger"
          "<div class=\"box-danger\" markdown=\"1\">#{title_html}#{content}</div>"
        else
        end
      end
    end
  
    class Details < Liquid::Block
      def initialize(tag_name, markup, tokens)
        super
        @detail_type, @summary, @open = markup.strip.split(" ", 3)
        @summary = @summary&.gsub(/^"(.*)"$/, '\1') 
        @open = @open == "open" ? true : false
      end
  
      def render(context)
        content = super
        case @detail_type
        when "block"
          details_html = "<details class=\"details-block\" markdown=\"1\""
        when "inline"
          details_html = "<details class=\"details-inline\" markdown=\"1\""
        else
          details_html = "<details markdown=\"1\""
        end
        details_html += " open" if @open
        details_html += "><summary>#{@summary}</summary>#{content}</details>"
      end
    end
  end
  
  Liquid::Template.register_tag('box', Jekyll::CustomTags)
  Liquid::Template.register_tag('details', Jekyll::Details)