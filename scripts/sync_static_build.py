import os

fp = r"apps/web/out/_next/static/chunks/app/analyst/page-4ffa10d0af1a3b3e.js"
with open(fp, "r", encoding="utf-8") as f:
    code = f.read()

target1 = '[h,b]=(0,r.useState)("SUM")'
repl1 = '[h,b]=(0,r.useState)((()=>{let q=(l.question||"").toLowerCase();if(q.includes("distinct")||q.includes("unique"))return "COUNT_DISTINCT";if(q.includes("how many")||q.includes("count")||q.includes("tier"))return "COUNT";if(q.includes("average")||q.includes("mean")||q.includes("avg"))return "AVG";return "SUM";})())'

target2 = "children:l.planned_calculation"
repl2 = 'children:((()=>{let m=p.trim(),gr=v.trim(),ag=h.toUpperCase(),flt=(g==="CUSTOM"&&_.trim())?("WHERE "+_.trim()):"",expr=ag==="COUNT"?"COUNT(*) AS observation_count":(ag==="COUNT_DISTINCT"?("COUNT(DISTINCT "+(m||"*")+") AS distinct_count"):(ag+"("+(m||"*")+") AS metric_val"));if(gr){let ord=ag==="COUNT"?"observation_count":(ag==="COUNT_DISTINCT"?"distinct_count":"metric_val");return "SELECT "+gr+", "+expr+"\\nFROM data\\n"+(flt?flt+"\\n":"")+"GROUP BY "+gr+"\\nORDER BY "+ord+" DESC;";}return "SELECT "+expr+"\\nFROM data"+(flt?"\\n"+flt:"")+";";})())'

target3 = 'children:[(0,a.jsx)("option",{value:"SUM",children:"SUM (Total)"}),(0,a.jsx)("option",{value:"AVG",children:"AVG (Mean Average)"}),(0,a.jsx)("option",{value:"COUNT",children:"COUNT (Rows / Observations)"}),(0,a.jsx)("option",{value:"MEDIAN",children:"MEDIAN (50th Percentile)"})]'
repl3 = 'children:[(0,a.jsx)("option",{value:"COUNT",children:"COUNT (Rows / Observations)"}),(0,a.jsx)("option",{value:"COUNT_DISTINCT",children:"COUNT_DISTINCT (Unique Entities)"}),(0,a.jsx)("option",{value:"SUM",children:"SUM (Total)"}),(0,a.jsx)("option",{value:"AVG",children:"AVG (Mean Average)"}),(0,a.jsx)("option",{value:"MEDIAN",children:"MEDIAN (50th Percentile)"}),(0,a.jsx)("option",{value:"MIN",children:"MIN (Minimum Value)"}),(0,a.jsx)("option",{value:"MAX",children:"MAX (Maximum Value)"})]'

target4 = "confirmed_assumptions:q?l.assumptions:[]"
repl4 = 'confirmed_assumptions:q?((()=>{let asm=[];let ag=h.toUpperCase();if(ag==="COUNT")asm.push("Metric: Total observation row count (COUNT(*))");else if(ag==="COUNT_DISTINCT")asm.push("Metric: Count of unique entities (COUNT(DISTINCT \'"+(p.trim()||"target")+"\'))");else asm.push("Metric: Quantitative "+ag+" of \'"+(p.trim()||"target")+"\'");if(v.trim())asm.push("Grouping dimension: \'"+v.trim()+"\'");else asm.push("Grouping: overall (no group dimension)");if(g==="CUSTOM"&&_.trim())asm.push("Scope filter: "+_.trim());else asm.push("Scope: All available records without filtering");for(let a of(l.assumptions||[])){let al=a.toLowerCase();if(!al.startsWith("metric:")&&!al.startsWith("grouping")&&!al.startsWith("scope:"))asm.push(a);}return asm;})()):[]'

if target1 in code:
    code = code.replace(target1, repl1, 1)
    print("Replaced target1")
else:
    print("target1 already replaced or not found")

if target2 in code:
    code = code.replace(target2, repl2, 1)
    print("Replaced target2")
else:
    print("target2 already replaced or not found")

if target3 in code:
    code = code.replace(target3, repl3, 1)
    print("Replaced target3")
else:
    print("target3 already replaced or not found")

if target4 in code:
    code = code.replace(target4, repl4, 1)
    print("Replaced target4")
else:
    print("target4 already replaced or not found")

with open(fp, "w", encoding="utf-8") as f:
    f.write(code)

print("Synchronization complete!")
