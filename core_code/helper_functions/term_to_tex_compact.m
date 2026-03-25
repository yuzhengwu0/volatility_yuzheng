function s = term_to_tex_compact(term)
    t = string(term);

    if contains(t, "Intercept")
        s = '\beta_0';
        return;
    end

    t = strrep(t, "b_{", "\beta_{");
    t = strrep(t, "b_",  "\beta_");

    t = strrep(t, "perf", "P");
    t = strrep(t, "corr", "C");
    t = strrep(t, "vol",  "V");
    t = strrep(t, "rt",   "R");

    s = char(t);
end